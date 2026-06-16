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

/**
 * Raw CUDA FP8 GEMM Kernel for Blackwell B200 (SM100a) — 2-CTA Cluster Mode
 *
 * No CUTLASS/CuTe dependencies — all PTX intrinsics are inlined.
 *
 * Implements:
 *   D[M,N] = A[M,K] * B[K,N]   (FP8 E4M3 inputs, BF16 output)
 *   with per-block UE8M0 scale factors (granularity 128)
 *
 * Architecture (cta_group::2):
 *   - 2-CTA cluster: 2 adjacent CTAs cooperate, sharing B data
 *   - Each CTA loads its own A tile + half of B tile
 *   - Leader CTA (rank 0) issues cta_group::2 MMA → uses both SMs' tensor cores
 *   - Both CTAs run epilogue independently (different output M tiles)
 *
 * Warp roles (per CTA):
 *   - Warp 0: TMA producer (loads A, B_half, SFA, SFB)
 *   - Warp 1: MMA consumer (leader CTA only) + UTCCP
 *   - Warp 2: UTCCP transposer (scale factor warp-transpose in SMEM)
 *   - Warps 4-7: Epilogue (TMEM→SMEM→Global via TMA store)
 *
 * Tile: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, 7 pipeline stages
 * LOAD_BLOCK_N=64 per CTA (half of B), UMMA_M=256 (cta_group::2)
 * Target: >=80% SOL on B200
 *
 * Compile: nvcc -gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -o fp8_gemm_2cta fp8_gemm_2cta.cu -lcuda -w
 */

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ============================================================================
// Configuration
// ============================================================================

// Tile dimensions
static constexpr uint32_t BLOCK_M = 128;
static constexpr uint32_t BLOCK_N = 128;
static constexpr uint32_t BLOCK_K = 128;

// 2-CTA cluster: each CTA loads half of B
static constexpr uint32_t NUM_MULTICAST = 2;
static constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / NUM_MULTICAST; // 64

// 7 stages fit with halved B SMEM (A=16K + B=8K per stage + SF)
static constexpr uint32_t NUM_STAGES = 7;

// FP8 scale factor granularity (128 = one SF per 128 K elements)
static constexpr uint32_t GRAN_K_A = 128;
static constexpr uint32_t GRAN_K_B = 128;

// Thread configuration: 128 non-epilogue + 128 epilogue = 256 total
static constexpr uint32_t NUM_NON_EPILOGUE_THREADS = 128;
static constexpr uint32_t NUM_EPILOGUE_THREADS = 128;
static constexpr uint32_t TOTAL_THREADS = NUM_NON_EPILOGUE_THREADS + NUM_EPILOGUE_THREADS;

// Layout
static constexpr uint32_t LAYOUT_AD_M = 128;
static constexpr uint32_t WAVE_BLOCK_M = (BLOCK_M < LAYOUT_AD_M) ? BLOCK_M : LAYOUT_AD_M;
static constexpr uint32_t NUM_M_WAVES = BLOCK_M / WAVE_BLOCK_M;

// Swizzle modes
static constexpr uint32_t SWIZZLE_A_MODE = 128;
static constexpr uint32_t SWIZZLE_B_MODE = 128;
static constexpr uint32_t SWIZZLE_CD_MODE = 128;

// Store dimensions — OUTPUT IS FP8 E4M3 (quantized in the epilogue).
// One BLOCK_N (=128) is exactly one 1x128 quant block per token row, so the per-block
// amax is a per-row reduction WITHIN a single CTA N-tile (no cross-tile reduction).
static constexpr uint32_t STORE_BLOCK_M = (BLOCK_M < LAYOUT_AD_M) ? BLOCK_M : LAYOUT_AD_M;
static constexpr uint32_t STORE_BLOCK_N = SWIZZLE_CD_MODE / sizeof(__nv_fp8_e4m3); // 128
static constexpr uint32_t NUM_STORES = BLOCK_N / STORE_BLOCK_N;                    // 1
static constexpr float E4M3_MAX = 448.0f;

// TMA store double-buffer
static constexpr uint32_t NUM_TMA_STORE_STAGES = 2;

// Scale factor alignment
static constexpr uint32_t UTCCP_ALIGNED_ELEMS = 128;
static constexpr uint32_t SF_BLOCK_M
    = ((BLOCK_M + UTCCP_ALIGNED_ELEMS - 1) / UTCCP_ALIGNED_ELEMS) * UTCCP_ALIGNED_ELEMS;
static constexpr uint32_t SF_BLOCK_N
    = ((BLOCK_N + UTCCP_ALIGNED_ELEMS - 1) / UTCCP_ALIGNED_ELEMS) * UTCCP_ALIGNED_ELEMS;

// SF stages per load
static constexpr uint32_t NUM_SFA_STAGES_PER_LOAD = (GRAN_K_A == 32) ? 1 : 4;
static constexpr uint32_t NUM_SFB_STAGES_PER_LOAD = (GRAN_K_B == 32) ? 1 : 4;

// SMEM sizes per stage (B is halved for 2-CTA)
static constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * SWIZZLE_CD_MODE;
static constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * NUM_TMA_STORE_STAGES;
static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);      // 16384
static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3); // 8192
static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);

// TMEM layout
static constexpr uint32_t NUM_SFA_TMEM_COLS = SF_BLOCK_M / 32;
static constexpr uint32_t NUM_SFB_TMEM_COLS = SF_BLOCK_N / 32;
static constexpr uint32_t NUM_EPILOGUE_STAGES
    = ((2 * NUM_M_WAVES * BLOCK_N + NUM_SFA_TMEM_COLS + NUM_SFB_TMEM_COLS) > 512) ? 1 : 2;
static constexpr uint32_t NUM_ACCUM_TMEM_COLS = NUM_EPILOGUE_STAGES * NUM_M_WAVES * BLOCK_N;
static constexpr uint32_t TMEM_START_COL_SFA = NUM_ACCUM_TMEM_COLS;
static constexpr uint32_t TMEM_START_COL_SFB = NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS;

__host__ __device__ constexpr uint32_t get_aligned_tmem_cols(uint32_t n)
{
    if (n <= 32)
        return 32;
    if (n <= 64)
        return 64;
    if (n <= 128)
        return 128;
    if (n <= 256)
        return 256;
    return 512;
}

static constexpr uint32_t NUM_TMEM_COLS
    = get_aligned_tmem_cols(NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS + NUM_SFB_TMEM_COLS);

// MMA instruction shape for cta_group::2
// UMMA_M = 256: both CTAs' 128 M rows combined
static constexpr uint32_t UMMA_M = LAYOUT_AD_M * NUM_MULTICAST; // 256
static constexpr uint32_t UMMA_N = BLOCK_N;                     // 128
static constexpr uint32_t UMMA_K = 32;

// Named barrier
static constexpr uint32_t NAMED_BARRIER_OFFSET = 8;

// Swizzle group size for L2-friendly scheduling — must be multiple of NUM_MULTICAST
static constexpr uint32_t SWIZZLE_GROUP_SIZE = 8;

// ============================================================================
// Utility functions
// ============================================================================

__device__ __forceinline__ uint32_t smem_to_uint(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t get_lane_idx()
{
    uint32_t lane;
    asm("mov.u32 %0, %laneid;" : "=r"(lane));
    return lane;
}

__device__ __forceinline__ uint32_t get_warp_idx_sync()
{
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
}

__device__ __forceinline__ uint32_t elect_one()
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

__host__ __device__ __forceinline__ uint32_t ceil_div(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__device__ __forceinline__ uint32_t block_rank_in_cluster()
{
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    return rank;
}

__device__ __forceinline__ void cluster_sync()
{
    asm volatile(
        "barrier.cluster.arrive;\n"
        "barrier.cluster.wait;\n" ::
            : "memory");
}

// ============================================================================
// Shared memory load/store helpers
// ============================================================================

__device__ __forceinline__ uint32_t ld_shared_u32(uint32_t const* ptr)
{
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

__device__ __forceinline__ void st_shared_u32(uint32_t const* ptr, uint32_t val)
{
    asm volatile("st.shared.u32 [%0], %1;" ::"l"(__cvta_generic_to_shared(ptr)), "r"(val));
}

__device__ __forceinline__ void st_shared_v4_u32(void const* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y),
        "r"(z), "r"(w));
}

// ============================================================================
// Barrier operations (mbarrier)
// ============================================================================

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count)
{
    uint32_t addr = smem_to_uint(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(count), "r"(addr));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t const* bar, uint32_t phase)
{
    uint32_t addr = smem_to_uint(bar);
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE;\n\t"
        "bra LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}" ::"r"(addr),
        "r"(phase), "r"(ticks));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t const* bar)
{
    uint32_t addr = smem_to_uint(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr));
}

// Cluster-scoped arrive: maps local SMEM address to CTA 0's address space
// Used by non-leader CTAs to arrive on leader's barrier
__device__ __forceinline__ void mbarrier_arrive_on_leader(uint64_t const* bar)
{
    uint32_t addr = smem_to_uint(bar);
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32 remAddr32, %0, 0;\n\t"
        "mbarrier.arrive.shared::cluster.b64 _, [remAddr32];\n\t"
        "}" ::"r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t const* bar, uint32_t tx_bytes)
{
    uint32_t addr = smem_to_uint(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;" ::"r"(tx_bytes), "r"(addr));
}

__device__ __forceinline__ void fence_barrier_init()
{
    asm volatile("fence.mbarrier_init.release.cluster;");
}

// ============================================================================
// tcgen05 fences
// ============================================================================

__device__ __forceinline__ void tcgen05_fence_before_thread_sync()
{
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync()
{
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// ============================================================================
// TMEM operations — cta_group::2
// ============================================================================

__device__ __forceinline__ void tmem_alloc_2sm(uint32_t* dst_smem, int num_cols)
{
    uint32_t addr = smem_to_uint(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(addr), "r"(num_cols));
}

__device__ __forceinline__ void tmem_free_2sm(uint32_t tmem_ptr, int num_cols)
{
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem_ptr), "r"(num_cols));
}

__device__ __forceinline__ void tmem_load_32x32b_x8(uint32_t addr, uint32_t& d0, uint32_t& d1, uint32_t& d2,
    uint32_t& d3, uint32_t& d4, uint32_t& d5, uint32_t& d6, uint32_t& d7)
{
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3), "=r"(d4), "=r"(d5), "=r"(d6), "=r"(d7)
                 : "r"(addr));
}

__device__ __forceinline__ void fence_tmem_load()
{
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

// ============================================================================
// UTCCP (SMEM -> TMEM copy) — cta_group::2
// ============================================================================

__device__ __forceinline__ void utccp_copy_2cta(uint64_t smem_desc, uint32_t tmem_col)
{
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" ::"r"(tmem_col), "l"(smem_desc));
}

// ============================================================================
// TMA operations
// ============================================================================

__device__ __forceinline__ void prefetch_tma_desc(void const* desc)
{
    uint64_t addr = reinterpret_cast<uint64_t>(desc);
    asm volatile("prefetch.tensormap [%0];" ::"l"(addr) : "memory");
}

__device__ __forceinline__ void tma_load_2d(void const* desc, uint64_t* mbar, void* smem, int32_t crd0, int32_t crd1)
{
    uint64_t gmem_desc = reinterpret_cast<uint64_t>(desc);
    uint32_t smem_mbar = smem_to_uint(mbar);
    uint32_t smem_ptr = smem_to_uint(smem);
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_ptr),
        "l"(gmem_desc), "r"(smem_mbar), "r"(crd0), "r"(crd1), "l"(cache_hint)
        : "memory");
}

__device__ __forceinline__ void tma_store_2d(void const* desc, void const* smem, int32_t crd0, int32_t crd1)
{
    uint64_t gmem_desc = reinterpret_cast<uint64_t>(desc);
    uint32_t smem_ptr = smem_to_uint(smem);
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" ::"l"(gmem_desc),
                 "r"(smem_ptr), "r"(crd0), "r"(crd1)
                 : "memory");
}

__device__ __forceinline__ void tma_store_fence()
{
    asm volatile("fence.proxy.async.shared::cta;");
}

__device__ __forceinline__ void tma_store_arrive()
{
    asm volatile("cp.async.bulk.commit_group;");
}

template <int Count>
__device__ __forceinline__ void tma_store_wait()
{
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(Count) : "memory");
}

__device__ __forceinline__ void fence_async_shared()
{
    asm volatile("fence.proxy.async.shared::cta;");
}

// ============================================================================
// tcgen05.commit — cta_group::2 multicast arrive
// ============================================================================

// For 2-CTA cluster: commit arrives on both CTAs' barriers via multicast
// MUST be called inside elect_one() — arrive::one generates 1 arrival per executing thread
__device__ __forceinline__ void umma_commit_arrive_2cta_multicast(uint64_t const* bar, uint16_t cta_mask)
{
    uint32_t addr = smem_to_uint(bar);
    if (elect_one())
    {
        asm volatile(
            "{\n\t"
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
            "}" ::"r"(addr),
            "h"(cta_mask));
    }
}

// For 2-CTA cluster: commit arrives on leader CTA's barrier only (no multicast)
__device__ __forceinline__ void umma_commit_arrive_2cta(uint64_t const* bar)
{
    uint32_t addr = smem_to_uint(bar);
    if (elect_one())
    {
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(addr));
    }
}

// ============================================================================
// tcgen05.mma FP8 block-scaled — cta_group::2
// ============================================================================

__device__ __forceinline__ void tcgen05_mma_fp8_block_scaled_2cta(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
    uint32_t scale_c, uint32_t instr_desc_hi, uint32_t tmem_sfa, uint32_t tmem_sfb)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t"
        "}\n" ::"r"(tmem_c),
        "l"(desc_a), "l"(desc_b), "r"(instr_desc_hi), "r"(scale_c), "r"(tmem_sfa), "r"(tmem_sfb));
}

// ============================================================================
// Named barrier
// ============================================================================

__device__ __forceinline__ void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id)
{
    uint32_t real_id = barrier_id + NAMED_BARRIER_OFFSET;
    asm volatile("bar.sync %0, %1;" ::"r"(real_id), "r"(num_threads));
}

// ============================================================================
// SMEM descriptor for UMMA (64-bit packed)
// ============================================================================

enum UmmaLayoutType : uint8_t
{
    LAYOUT_SWIZZLE_NONE = 0,
    LAYOUT_SWIZZLE_128B = 2,
    LAYOUT_SWIZZLE_64B = 4,
    LAYOUT_SWIZZLE_32B = 6,
};

struct SmemDesc
{
    uint32_t lo;
    uint32_t hi;

    __device__ __forceinline__ operator uint64_t() const
    {
        return (uint64_t(hi) << 32) | uint64_t(lo);
    }
};

__device__ __forceinline__ SmemDesc make_smem_desc_pair(
    UmmaLayoutType layout, void* smem_ptr, uint32_t stride_byte_offset, uint32_t leading_byte_offset)
{
    uint64_t desc = 0;
    uint32_t uint_ptr = smem_to_uint(smem_ptr);
    desc |= uint64_t(uint_ptr >> 4) & 0x3FFF;
    desc |= (uint64_t(leading_byte_offset >> 4) & 0x3FFF) << 16;
    desc |= (uint64_t(stride_byte_offset >> 4) & 0x3FFF) << 32;
    desc |= uint64_t(1) << 46; // version = 1
    desc |= uint64_t(layout) << 61;
    SmemDesc s;
    s.lo = uint32_t(desc);
    s.hi = uint32_t(desc >> 32);
    return s;
}

__device__ __forceinline__ SmemDesc make_umma_desc_k_major_fp8(void* base_smem_ptr)
{
    return make_smem_desc_pair(LAYOUT_SWIZZLE_128B, base_smem_ptr, 8 * BLOCK_K * 1, 0);
}

__device__ __forceinline__ SmemDesc make_sf_desc_pair(void* smem_ptr)
{
    return make_smem_desc_pair(LAYOUT_SWIZZLE_NONE, smem_ptr, 8 * 16, 0);
}

__device__ __forceinline__ void replace_desc_addr(SmemDesc& desc, void* smem_ptr)
{
    uint32_t uint_ptr = smem_to_uint(smem_ptr);
    uint32_t new_addr = (uint_ptr >> 4) & 0x3FFF;
    desc.lo = (desc.lo & ~0x3FFFu) | new_addr;
}

// ============================================================================
// Instruction descriptor for FP8 block-scaled MMA (cta_group::2)
// UMMA_M=256, UMMA_N=128
// ============================================================================

__host__ __device__ constexpr uint32_t make_fp8_block_scaled_instr_desc_2cta()
{
    uint32_t desc = 0;
    // a_format = E4M3 = 0, b_format = E4M3 = 0
    // a_major = K = 0, b_major = K = 0
    desc |= ((UMMA_N / 8) & 0x3F) << 17;  // n_dim = 128/8 = 16
    desc |= 1u << 23;                     // scale_format = UE8M0
    desc |= ((UMMA_M / 16) & 0x1F) << 24; // m_dim = 256/16 = 16
    return desc;
}

__device__ __forceinline__ uint32_t set_sf_ids(uint32_t base_desc, uint32_t sfa_id, uint32_t sfb_id)
{
    uint32_t d = base_desc;
    d = (d & ~(3u << 4)) | ((sfb_id & 3u) << 4);
    d = (d & ~(3u << 29)) | ((sfa_id & 3u) << 29);
    return d;
}

// ============================================================================
// Persistent scheduler for 2-CTA cluster
// Swizzle ensures adjacent blockIdx values get adjacent M blocks, same N block
// ============================================================================

struct PersistentScheduler2CTA
{
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;
    uint32_t num_blocks;
    uint32_t num_sms;
    int current_iter;
    uint32_t current_shape_k;

    __device__ __forceinline__ void init(uint32_t M, uint32_t N, uint32_t K, uint32_t nsms)
    {
        num_m_blocks = ceil_div(M, BLOCK_M);
        num_n_blocks = ceil_div(N, BLOCK_N);
        num_blocks = num_m_blocks * num_n_blocks;
        num_sms = nsms; // total CTAs (= 2x number of clusters)
        current_iter = -1;
        current_shape_k = K;
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block, uint32_t& n_block)
    {
        uint32_t block_idx = (++current_iter) * num_sms + blockIdx.x;
        if (block_idx >= num_blocks)
            return false;

        // L2-friendly swizzle: group M blocks, iterate N within
        // For 2-CTA: adjacent blockIdx values get adjacent M blocks within same N column
        // This ensures cluster CTAs (adjacent blockIdx) work on same N, adjacent M
        constexpr uint32_t GROUP_SIZE = SWIZZLE_GROUP_SIZE; // must be multiple of NUM_MULTICAST
        uint32_t num_groups = ceil_div(num_m_blocks, GROUP_SIZE);
        uint32_t blocks_per_group = num_n_blocks * GROUP_SIZE;
        uint32_t group_idx = block_idx / blocks_per_group;
        uint32_t in_group = block_idx % blocks_per_group;

        uint32_t actual_group_size = GROUP_SIZE;
        if (group_idx == num_groups - 1)
        {
            uint32_t remaining = num_m_blocks - group_idx * GROUP_SIZE;
            if (remaining > 0)
                actual_group_size = remaining;
        }

        // Within group: iterate M first for cluster locality, then N
        m_block = group_idx * GROUP_SIZE + in_group % actual_group_size;
        n_block = in_group / actual_group_size;

        // Clamp
        if (m_block >= num_m_blocks)
        {
            m_block = 0;
            n_block = 0;
        }
        if (n_block >= num_n_blocks)
        {
            m_block = 0;
            n_block = 0;
        }

        return true;
    }
};

// ============================================================================
// BF16 conversion helper
// ============================================================================

__device__ __forceinline__ uint32_t fp32x2_to_bf16x2(uint32_t a, uint32_t b)
{
    __nv_bfloat162 bf = __float22bfloat162_rn({*reinterpret_cast<float*>(&a), *reinterpret_cast<float*>(&b)});
    return *reinterpret_cast<uint32_t*>(&bf);
}

// Quantize 4 fp32 (passed as raw bits) → 4 packed E4M3 bytes (one uint32), scaled by `inv`.
__device__ __forceinline__ uint32_t quant4_e4m3(uint32_t a, uint32_t b, uint32_t c, uint32_t d, float inv)
{
    unsigned short lo = __nv_cvt_float2_to_fp8x2(
        make_float2(__uint_as_float(a) * inv, __uint_as_float(b) * inv), __NV_SATFINITE, __NV_E4M3);
    unsigned short hi = __nv_cvt_float2_to_fp8x2(
        make_float2(__uint_as_float(c) * inv, __uint_as_float(d) * inv), __NV_SATFINITE, __NV_E4M3);
    return (uint32_t(hi) << 16) | uint32_t(lo);
}

// ============================================================================
// MAIN KERNEL — 2-CTA Cluster
// ============================================================================

__global__ void __launch_bounds__(TOTAL_THREADS, 1) fp8_gemm_kernel_2cta(uint32_t shape_m, uint32_t shape_n,
    uint32_t shape_k, const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b, const __grid_constant__ CUtensorMap tensor_map_sfa,
    const __grid_constant__ CUtensorMap tensor_map_sfb, const __grid_constant__ CUtensorMap tensor_map_d,
    uint8_t* __restrict__ sf_out, // per-(token, 128-N-block) UE8M0 scale byte, [M, N/128] row-major
    uint32_t num_sms)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    const uint32_t warp_idx = get_warp_idx_sync();
    const uint32_t lane_idx = get_lane_idx();
    const uint32_t cta_rank = block_rank_in_cluster();
    bool const is_leader_cta = (cta_rank == 0);

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // ===== SMEM layout ===== (CD region now staged as FP8 bytes)
    auto smem_cd_ptr = [&](uint32_t stage) -> uint8_t* { return smem_buffer + stage * SMEM_CD_SIZE_PER_STAGE; };
    auto smem_a_ptr = [&](uint32_t stage) -> __nv_fp8_e4m3*
    { return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_CD_SIZE + stage * SMEM_A_SIZE_PER_STAGE); };
    auto smem_b_ptr = [&](uint32_t stage) -> __nv_fp8_e4m3*
    {
        return reinterpret_cast<__nv_fp8_e4m3*>(
            smem_buffer + SMEM_CD_SIZE + NUM_STAGES * SMEM_A_SIZE_PER_STAGE + stage * SMEM_B_SIZE_PER_STAGE);
    };
    uint8_t* sf_start = smem_buffer + SMEM_CD_SIZE + NUM_STAGES * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa_ptr = [&](uint32_t stage) -> uint32_t*
    { return reinterpret_cast<uint32_t*>(sf_start + stage * SMEM_SFA_SIZE_PER_STAGE); };
    auto smem_sfb_ptr = [&](uint32_t stage) -> uint32_t*
    {
        return reinterpret_cast<uint32_t*>(
            sf_start + NUM_STAGES * SMEM_SFA_SIZE_PER_STAGE + stage * SMEM_SFB_SIZE_PER_STAGE);
    };

    // Barriers: [full × NUM_STAGES, empty × NUM_STAGES, sf_full × NUM_STAGES,
    //            tmem_full × NUM_EPILOGUE_STAGES, tmem_empty × NUM_EPILOGUE_STAGES]
    auto bar_start
        = reinterpret_cast<uint64_t*>(sf_start + NUM_STAGES * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
    auto full_barrier = [&](uint32_t i) -> uint64_t* { return bar_start + i; };
    auto empty_barrier = [&](uint32_t i) -> uint64_t* { return bar_start + NUM_STAGES + i; };
    auto sf_full_barrier = [&](uint32_t i) -> uint64_t* { return bar_start + NUM_STAGES * 2 + i; };
    auto tmem_full_barrier = [&](uint32_t i) -> uint64_t* { return bar_start + NUM_STAGES * 3 + i; };
    auto tmem_empty_barrier
        = [&](uint32_t i) -> uint64_t* { return bar_start + NUM_STAGES * 3 + NUM_EPILOGUE_STAGES + i; };

    // TMEM pointer in shared memory
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(bar_start + NUM_STAGES * 3 + NUM_EPILOGUE_STAGES * 2);

    // ===== Prefetch TMA descriptors =====
    if (warp_idx == 0 && elect_one())
    {
        prefetch_tma_desc(&tensor_map_a);
        prefetch_tma_desc(&tensor_map_b);
        prefetch_tma_desc(&tensor_map_sfa);
        prefetch_tma_desc(&tensor_map_sfb);
        prefetch_tma_desc(&tensor_map_d);
    }

    // ===== Initialize barriers =====
    if (warp_idx == 1 && elect_one())
    {
        for (uint32_t i = 0; i < NUM_STAGES; i++)
        {
            mbarrier_init(full_barrier(i), 1);
            mbarrier_init(empty_barrier(i), 1);
            // SF full: UTCCP warp (32 threads) × NUM_MULTICAST CTAs arrive
            mbarrier_init(sf_full_barrier(i), NUM_MULTICAST * 32);
        }
        for (uint32_t i = 0; i < NUM_EPILOGUE_STAGES; i++)
        {
            mbarrier_init(tmem_full_barrier(i), 1);
            // TMEM empty: epilogue threads × NUM_MULTICAST CTAs
            mbarrier_init(tmem_empty_barrier(i), NUM_MULTICAST * STORE_BLOCK_M);
        }
        fence_barrier_init();
    }
    else if (warp_idx == 2)
    {
        // Allocate TMEM — cta_group::2
        tmem_alloc_2sm(tmem_ptr_in_smem, NUM_TMEM_COLS);
    }

    // Cluster sync after barrier init + TMEM alloc
    cluster_sync();

    // Verify TMEM starts at 0
    // (2-CTA alloc distributes TMEM coherently)

    // ===== Scheduler =====
    PersistentScheduler2CTA scheduler;
    scheduler.init(shape_m, shape_n, shape_k, num_sms);
    uint32_t m_block_idx, n_block_idx;

    // ===== Pipeline state =====
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx)
    {
        ++k_block_idx;
        stage_idx = (stage_idx == NUM_STAGES - 1) ? 0 : stage_idx + 1;
        phase ^= (stage_idx == 0);
    };

    constexpr uint16_t CTA_MASK = (1 << NUM_MULTICAST) - 1; // 0x3 for 2-CTA

    // ========================================================================
    // WARP 0: TMA PRODUCER
    // Both CTAs issue TMA loads independently
    // ========================================================================
    if (warp_idx == 0 && elect_one())
    {
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            uint32_t num_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx))
            {
                // Wait for consumer to release this stage
                mbarrier_wait(empty_barrier(stage_idx), phase ^ 1);

                uint32_t m_idx = m_block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;

                // N index: each CTA gets different half of B
                // CTA 0 loads B[n_block*BLOCK_N : n_block*BLOCK_N + LOAD_BLOCK_N]
                // CTA 1 loads B[n_block*BLOCK_N + LOAD_BLOCK_N : (n_block+1)*BLOCK_N]
                uint32_t n_idx = n_block_idx * BLOCK_N + cta_rank * LOAD_BLOCK_N;

                // TMA load A (K-major: inner=K, outer=M) — each CTA loads full A
                tma_load_2d(&tensor_map_a, reinterpret_cast<uint64_t*>(full_barrier(stage_idx)), smem_a_ptr(stage_idx),
                    k_idx, m_idx);

                // TMA load B (K-major: inner=K, outer=N) — each CTA loads half B
                tma_load_2d(&tensor_map_b, reinterpret_cast<uint64_t*>(full_barrier(stage_idx)), smem_b_ptr(stage_idx),
                    k_idx, n_idx);

                uint32_t arrival_bytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;

                // TMA load scale factors
                if (k_block_idx % NUM_SFA_STAGES_PER_LOAD == 0)
                {
                    uint32_t sfa_k_idx = k_idx / (BLOCK_K * NUM_SFA_STAGES_PER_LOAD);
                    tma_load_2d(&tensor_map_sfa, reinterpret_cast<uint64_t*>(full_barrier(stage_idx)),
                        smem_sfa_ptr(stage_idx), m_block_idx * BLOCK_M, sfa_k_idx);
                    arrival_bytes += BLOCK_M * sizeof(uint32_t);
                }
                if (k_block_idx % NUM_SFB_STAGES_PER_LOAD == 0)
                {
                    uint32_t sfb_k_idx = k_idx / (BLOCK_K * NUM_SFB_STAGES_PER_LOAD);
                    // Both CTAs load full BLOCK_N SFB (MMA needs all N scale factors)
                    tma_load_2d(&tensor_map_sfb, reinterpret_cast<uint64_t*>(full_barrier(stage_idx)),
                        smem_sfb_ptr(stage_idx), n_block_idx * BLOCK_N, sfb_k_idx);
                    arrival_bytes += BLOCK_N * sizeof(uint32_t);
                }

                mbarrier_arrive_expect_tx(full_barrier(stage_idx), arrival_bytes);
            }
        }
    }
    // ========================================================================
    // WARP 1: MMA CONSUMER — LEADER CTA ONLY
    // ========================================================================
    else if (warp_idx == 1 && is_leader_cta)
    {
        constexpr uint32_t BASE_INSTR_DESC = make_fp8_block_scaled_instr_desc_2cta();

        // Build base SMEM descriptors
        SmemDesc a_desc = make_umma_desc_k_major_fp8(smem_a_ptr(0));
        SmemDesc b_desc = make_umma_desc_k_major_fp8(smem_b_ptr(0));
        SmemDesc sf_desc = make_sf_desc_pair(nullptr);

        // Pre-compute per-stage descriptor offsets in warp lanes
        uint32_t a_desc_lo = (lane_idx < NUM_STAGES) ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = (lane_idx < NUM_STAGES) ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            // Wait for TMEM to be freed by epilogue
            uint32_t accum_stage = scheduler.current_iter % NUM_EPILOGUE_STAGES;
            uint32_t accum_phase = (scheduler.current_iter / NUM_EPILOGUE_STAGES) & 1;
            mbarrier_wait(tmem_empty_barrier(accum_stage), accum_phase ^ 1);
            tcgen05_fence_after_thread_sync();

            // Lambda for umma arrive on empty/tmem barriers
            auto empty_barrier_arrive = [&](bool do_tmem_full_arrive)
            {
                // Multicast arrive: signal both CTAs' empty barrier
                umma_commit_arrive_2cta_multicast(empty_barrier(stage_idx), CTA_MASK);
                if (do_tmem_full_arrive)
                    umma_commit_arrive_2cta_multicast(tmem_full_barrier(accum_stage), CTA_MASK);
            };

            uint32_t num_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx))
            {
                uint32_t sfa_stage_in_group = k_block_idx % NUM_SFA_STAGES_PER_LOAD;
                uint32_t sfb_stage_in_group = k_block_idx % NUM_SFB_STAGES_PER_LOAD;

                // Wait for SF transpose to complete (warp 2 handles transpose)
                mbarrier_wait(sf_full_barrier(stage_idx), phase);
                tcgen05_fence_after_thread_sync();

                // UTCCP: copy transposed scale factors from SMEM to TMEM
                if (k_block_idx % NUM_SFA_STAGES_PER_LOAD == 0 && elect_one())
                {
                    for (uint32_t i = 0; i < SF_BLOCK_M / UTCCP_ALIGNED_ELEMS; i++)
                    {
                        replace_desc_addr(sf_desc, smem_sfa_ptr(stage_idx) + i * UTCCP_ALIGNED_ELEMS);
                        utccp_copy_2cta(uint64_t(sf_desc), TMEM_START_COL_SFA + i * 4);
                    }
                }
                if (k_block_idx % NUM_SFB_STAGES_PER_LOAD == 0 && elect_one())
                {
                    for (uint32_t i = 0; i < SF_BLOCK_N / UTCCP_ALIGNED_ELEMS; i++)
                    {
                        replace_desc_addr(sf_desc, smem_sfb_ptr(stage_idx) + i * UTCCP_ALIGNED_ELEMS);
                        utccp_copy_2cta(uint64_t(sf_desc), TMEM_START_COL_SFB + i * 4);
                    }
                }
                __syncwarp();

                // Issue MMA — cta_group::2
                uint32_t a_base_lo = __shfl_sync(0xffffffff, a_desc_lo, stage_idx);
                uint32_t b_base_lo = __shfl_sync(0xffffffff, b_desc_lo, stage_idx);

                if (elect_one())
                {
#pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; k++)
                    {
                        uint32_t sfa_id = (GRAN_K_A == 32) ? k : sfa_stage_in_group;
                        uint32_t sfb_id = (GRAN_K_B == 32) ? k : sfb_stage_in_group;
                        uint32_t instr = set_sf_ids(BASE_INSTR_DESC, sfa_id, sfb_id);

                        // Advance B descriptor for K sub-tile
                        b_desc.lo = b_base_lo + (k * UMMA_K * 1) / 16;

#pragma unroll
                        for (uint32_t w = 0; w < NUM_M_WAVES; w++)
                        {
                            // Advance A descriptor for K sub-tile and M wave
                            a_desc.lo = a_base_lo + (w * WAVE_BLOCK_M * BLOCK_K + k * UMMA_K * 1) / 16;

                            tcgen05_mma_fp8_block_scaled_2cta(a_desc, b_desc,
                                accum_stage * NUM_M_WAVES * BLOCK_N + w * BLOCK_N, k_block_idx > 0 || k > 0,
                                (instr << 0), TMEM_START_COL_SFA + w * (UTCCP_ALIGNED_ELEMS / 32), TMEM_START_COL_SFB);
                        }
                    }
                }

                // Commit: signal empty barrier (multicast to both CTAs)
                empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
            }
        }

        // Extra wait for safe barrier destruction in 2-CTA mode
        auto const& iter_idx = scheduler.current_iter - 1;
        if (iter_idx >= 0)
        {
            auto const& accum_phase_idx = (iter_idx / NUM_EPILOGUE_STAGES) & 1;
            mbarrier_wait(tmem_empty_barrier(iter_idx % NUM_EPILOGUE_STAGES), accum_phase_idx);
        }
    }
    // ========================================================================
    // WARP 2: UTCCP TRANSPOSER (both CTAs)
    // ========================================================================
    else if (warp_idx == 2)
    {
        // Warp transpose for UTCCP: in-place transpose in SMEM using XOR trick
        auto utccp_warp_transpose = [&](uint32_t const* smem_ptr)
        {
            uint32_t values[4];
#pragma unroll
            for (uint32_t i = 0; i < 4; i++)
                values[i] = ld_shared_u32(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
#pragma unroll
            for (uint32_t i = 0; i < 4; i++)
                st_shared_u32(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            uint32_t num_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx))
            {
                // Wait for TMA to deliver data (A, B, SF)
                mbarrier_wait(full_barrier(stage_idx), phase);

                // Transpose SFA in SMEM (both CTAs)
                if (k_block_idx % NUM_SFA_STAGES_PER_LOAD == 0)
                {
                    for (uint32_t i = 0; i < SF_BLOCK_M / UTCCP_ALIGNED_ELEMS; i++)
                        utccp_warp_transpose(smem_sfa_ptr(stage_idx) + i * UTCCP_ALIGNED_ELEMS);
                    fence_async_shared();
                }

                // Transpose SFB in SMEM (both CTAs)
                if (k_block_idx % NUM_SFB_STAGES_PER_LOAD == 0)
                {
                    for (uint32_t i = 0; i < SF_BLOCK_N / UTCCP_ALIGNED_ELEMS; i++)
                        utccp_warp_transpose(smem_sfb_ptr(stage_idx) + i * UTCCP_ALIGNED_ELEMS);
                    fence_async_shared();
                }

                // Signal MMA that SF data is ready — arrive on LEADER CTA's barrier
                // Both CTAs' warp 2 arrive here; leader CTA's MMA waits on this
                mbarrier_arrive_on_leader(sf_full_barrier(stage_idx));
            }
        }
    }
    // ========================================================================
    // WARPS 4-7: EPILOGUE (both CTAs independently)
    // ========================================================================
    else if (warp_idx >= NUM_NON_EPILOGUE_THREADS / 32 && warp_idx < (NUM_NON_EPILOGUE_THREADS + STORE_BLOCK_M) / 32)
    {
        const uint32_t epilogue_warp_idx = warp_idx - (NUM_NON_EPILOGUE_THREADS / 32);

        constexpr uint32_t BANK_GROUP_BYTES = 16;
        constexpr uint32_t ELEMS_PER_BANK_GROUP = BANK_GROUP_BYTES / sizeof(__nv_fp8_e4m3); // 16 fp8
        constexpr uint32_t NUM_BANK_GROUPS = STORE_BLOCK_N / ELEMS_PER_BANK_GROUP;          // 8
        // NUM_M_WAVES == 1 and NUM_STORES == 1 for this fp8 config (STORE_BLOCK_N == BLOCK_N).
        const uint32_t num_n_blocks = shape_n / STORE_BLOCK_N; // == N/128
        uint32_t tma_stage_idx = 0;

        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            uint32_t accum_stage = scheduler.current_iter % NUM_EPILOGUE_STAGES;
            mbarrier_wait(tmem_full_barrier(accum_stage), (scheduler.current_iter / NUM_EPILOGUE_STAGES) & 1);
            tcgen05_fence_after_thread_sync();

            const uint32_t accum_base = accum_stage * NUM_M_WAVES * BLOCK_N; // NUM_M_WAVES == 1
            const uint32_t row = lane_idx; // this thread owns one output row within its 32-wide band

            // ---- PASS 1: per-row amax over the full 128-wide quant block (this CTA N-tile).
            //      Each lane owns one row, so this is a pure per-thread reduction — no cross-
            //      thread / cross-tile communication (this is exactly what makes the fuse free). ----
            float amax = 0.0f;
#pragma unroll
            for (uint32_t i = 0; i < NUM_BANK_GROUPS; i++)
            {
#pragma unroll
                for (uint32_t h = 0; h < 2; h++)
                {
                    uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                    tmem_load_32x32b_x8(accum_base + i * ELEMS_PER_BANK_GROUP + h * 8, v0, v1, v2, v3, v4, v5, v6, v7);
                    fence_tmem_load();
                    amax = fmaxf(amax,
                        fmaxf(fmaxf(fmaxf(fabsf(__uint_as_float(v0)), fabsf(__uint_as_float(v1))),
                                  fmaxf(fabsf(__uint_as_float(v2)), fabsf(__uint_as_float(v3)))),
                            fmaxf(fmaxf(fabsf(__uint_as_float(v4)), fabsf(__uint_as_float(v5))),
                                fmaxf(fabsf(__uint_as_float(v6)), fabsf(__uint_as_float(v7))))));
                }
            }
            // ---- UE8M0 scale, byte-exact with production fp8_quantize_1x128_packed_ue8m0:
            //      __nv_cvt_float_to_e8m0(amax/448, SATFINITE, RoundPosInf) (round toward +Inf).
            //      The e8m0 byte is the biased exponent (bias 127); quant scale = 2^(127-byte). ----
            amax = fmaxf(amax, 1e-10f);
            const uint8_t sf_byte = __nv_cvt_float_to_e8m0(amax * (1.0f / E4M3_MAX), __NV_SATFINITE, cudaRoundPosInf);
            float const inv = (sf_byte == 0) ? 1.0f : exp2f(127.0f - (float) sf_byte); // 2^-e (matches prod)
            const uint32_t token = m_block_idx * BLOCK_M + epilogue_warp_idx * 32 + lane_idx;
            if (token < shape_m)
            {
                // Packed A-scale layout deep_gemm's fp8_gemm_nt reads (from fp8Quantize.cpp +
                // fp8_blockscale_quant_packed.cu): physical [num_packed_sf_k, m_aligned] int32,
                // viewed (m, num_packed_sf_k) stride (1, m_aligned). int32 at [n_block/4, token]
                // holds 4 UE8M0 bytes (b0|b1<<8|..), byte (n_block%4) = this n_block. m_aligned=ceil(M/4)*4.
                const uint32_t m_aligned = (shape_m + 3u) / 4u * 4u;
                const uint32_t kp = n_block_idx >> 2;   // n_block / 4
                const uint32_t bsel = n_block_idx & 3u; // n_block % 4
                sf_out[((size_t) kp * m_aligned + token) * 4u + bsel] = sf_byte;
            }

            // Make sure the TMA store from 2 tiles ago has drained before reusing this SMEM stage.
            tma_store_wait<NUM_TMA_STORE_STAGES - 1>();

// ---- PASS 2: re-read TMEM, quantize → e4m3, swizzled SMEM store. ----
#pragma unroll
            for (uint32_t i = 0; i < NUM_BANK_GROUPS; i++)
            {
                uint32_t v0, v1, v2, v3, v4, v5, v6, v7, q0, q1, q2, q3, q4, q5, q6, q7;
                tmem_load_32x32b_x8(accum_base + i * ELEMS_PER_BANK_GROUP, v0, v1, v2, v3, v4, v5, v6, v7);
                fence_tmem_load();
                tmem_load_32x32b_x8(accum_base + i * ELEMS_PER_BANK_GROUP + 8, q0, q1, q2, q3, q4, q5, q6, q7);
                fence_tmem_load();
                uint32_t col = i ^ (row % (SWIZZLE_CD_MODE / 16));
                uint8_t* smem_ptr = smem_cd_ptr(tma_stage_idx) + epilogue_warp_idx * 32 * SWIZZLE_CD_MODE
                    + row * (BANK_GROUP_BYTES * 8) + col * BANK_GROUP_BYTES;
                st_shared_v4_u32(smem_ptr, quant4_e4m3(v0, v1, v2, v3, inv), quant4_e4m3(v4, v5, v6, v7, inv),
                    quant4_e4m3(q0, q1, q2, q3, inv), quant4_e4m3(q4, q5, q6, q7, inv));
            }

            tcgen05_fence_before_thread_sync();
            mbarrier_arrive_on_leader(tmem_empty_barrier(accum_stage));

            __syncwarp();
            tma_store_fence();
            named_barrier_sync(STORE_BLOCK_M, 0);
            if (epilogue_warp_idx == 0 && elect_one())
            {
                tma_store_2d(&tensor_map_d, smem_cd_ptr(tma_stage_idx), n_block_idx * BLOCK_N, m_block_idx * BLOCK_M);
                tma_store_arrive();
            }
            tma_stage_idx = (tma_stage_idx + 1) % NUM_TMA_STORE_STAGES;
        }

        // Deallocate TMEM — cta_group::2
        if (epilogue_warp_idx == STORE_BLOCK_M / 32 - 1)
            tmem_free_2sm(0, NUM_TMEM_COLS);
    }
#endif
}

// ============================================================================
// HOST-SIDE: TMA Descriptor creation + Launcher
// ============================================================================

void check_cuda(CUresult err, char const* msg)
{
    if (err != CUDA_SUCCESS)
    {
        char const* str;
        cuGetErrorString(err, &str);
        fprintf(stderr, "CUDA Driver Error: %s - %s\n", msg, str);
        exit(1);
    }
}

void check_cuda_rt(cudaError_t err, char const* msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

CUtensorMap create_tma_desc_2d(void* gmem_ptr, CUtensorMapDataType dtype, uint64_t dim0, uint64_t dim1,
    uint64_t stride1_bytes, uint32_t box0, uint32_t box1, CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B)
{
    CUtensorMap map;
    uint64_t dims[2] = {dim0, dim1};
    uint64_t strides[1] = {stride1_bytes};
    uint32_t box_dims[2] = {box0, box1};
    uint32_t elem_strides[2] = {1, 1};

    check_cuda(
        cuTensorMapEncodeTiled(&map, dtype, 2, gmem_ptr, dims, strides, box_dims, elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled");

    return map;
}

struct GemmProblem
{
    uint32_t M, N, K;
    __nv_fp8_e4m3* A;
    __nv_fp8_e4m3* B;
    __nv_fp8_e4m3* D; // FP8 output (quantized in the epilogue)
    uint8_t* sf;      // per-(token, 128-N-block) output UE8M0 scale byte, [M, N/128]
    uint32_t* sfa;
    uint32_t* sfb;
};

void launch_fp8_gemm_2cta(GemmProblem& prob, cudaStream_t stream = 0)
{
    uint32_t M = prob.M, N = prob.N, K = prob.K;

    // One-time per-process setup (device caps, SMEM attr, num_sms). Cached so the per-call
    // path holds NO cudaFuncSetAttribute / device queries — only host-side tensor-map builds
    // and cudaLaunchKernelEx, which are safe to capture inside a CUDA graph.
    static bool setup_done = false;
    static uint32_t s_num_sms = 0;
    static uint32_t s_smem_size = 0;
    if (!setup_done)
    {
        uint32_t smem_size = SMEM_CD_SIZE + NUM_STAGES * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE)
            + NUM_STAGES * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE)
            + (NUM_STAGES * 3 + NUM_EPILOGUE_STAGES * 2) * sizeof(uint64_t) + sizeof(uint32_t);
        assert(smem_size <= 232448);

        int device;
        cudaGetDevice(&device);
        int major, minor;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        if (major < 10)
        {
            fprintf(stderr, "Error: This kernel requires SM100a (compute capability 10.0a), got %d.%d\n", major, minor);
            exit(1);
        }
        int num_sms_int;
        cudaDeviceGetAttribute(&num_sms_int, cudaDevAttrMultiProcessorCount, device);
        uint32_t num_sms = static_cast<uint32_t>(num_sms_int);
        num_sms = (num_sms / NUM_MULTICAST) * NUM_MULTICAST;
        if (num_sms == 0)
        {
            fprintf(stderr, "Error: Need at least %u SMs for cluster size %u\n", NUM_MULTICAST, NUM_MULTICAST);
            exit(1);
        }
        int max_cluster_size = 0;
        cudaDeviceGetAttribute(&max_cluster_size, cudaDevAttrClusterLaunch, device);
        if (!max_cluster_size)
        {
            fprintf(stderr, "Error: Device does not support cluster launch (cudaDevAttrClusterLaunch=0)\n");
            exit(1);
        }
        check_cuda_rt(
            cudaFuncSetAttribute(fp8_gemm_kernel_2cta, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size),
            "set smem size");
        check_cuda_rt(cudaFuncSetAttribute(fp8_gemm_kernel_2cta, cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
            "set non-portable cluster size");
        printf("SMEM required: %u bytes (max 232448)\n", smem_size);
        printf("Grid: %u CTAs (%u clusters of %u), from %d SMs\n", num_sms, num_sms / NUM_MULTICAST, NUM_MULTICAST,
            num_sms_int);
        s_num_sms = num_sms;
        s_smem_size = smem_size;
        setup_done = true;
    }
    const uint32_t num_sms = s_num_sms;
    const uint32_t smem_size = s_smem_size;

    // TMA descriptors
    // A: [K, M] K-major, inner=K, outer=M
    CUtensorMap tma_a
        = create_tma_desc_2d(prob.A, CU_TENSOR_MAP_DATA_TYPE_UINT8, K, M, (uint64_t) K * 1, BLOCK_K, BLOCK_M);

    // B: [K, N] K-major, inner=K, outer=N
    // For 2-CTA: each CTA loads LOAD_BLOCK_N (half) of B
    CUtensorMap tma_b = create_tma_desc_2d(
        prob.B, CU_TENSOR_MAP_DATA_TYPE_UINT8, K, N, (uint64_t) K * 1, BLOCK_K, LOAD_BLOCK_N); // box = [128, 64]

    // SFA: [sfa_k, m_aligned] M-contiguous. The activation scale buffer (from
    // _fp8_quantize_1x128_ue8m0 / fp8_quantize_1x128_packed_ue8m0) pads its leading dim to
    // m_aligned = ceil(M/4)*4, so the K-block stride is m_aligned (NOT M) for arbitrary M.
    uint32_t sfa_k = ceil_div(K, GRAN_K_A * 4);
    uint32_t m_aligned_sf = (M + 3) / 4 * 4;
    CUtensorMap tma_sfa = create_tma_desc_2d(prob.sfa, CU_TENSOR_MAP_DATA_TYPE_UINT32, M, sfa_k,
        (uint64_t) m_aligned_sf * sizeof(uint32_t), BLOCK_M, 1, CU_TENSOR_MAP_SWIZZLE_NONE);

    // SFB: [sfb_k, N] N-contiguous — both CTAs load full BLOCK_N
    uint32_t sfb_k = ceil_div(K, GRAN_K_B * 4);
    CUtensorMap tma_sfb = create_tma_desc_2d(prob.sfb, CU_TENSOR_MAP_DATA_TYPE_UINT32, N, sfb_k,
        (uint64_t) N * sizeof(uint32_t), BLOCK_N, 1, CU_TENSOR_MAP_SWIZZLE_NONE);

    // D: [N, M] for TMA store (inner=N, outer=M) — FP8 E4M3 output
    CUtensorMap tma_d = create_tma_desc_2d(prob.D, CU_TENSOR_MAP_DATA_TYPE_UINT8, N, M,
        (uint64_t) N * sizeof(__nv_fp8_e4m3), STORE_BLOCK_N, STORE_BLOCK_M);

    // Launch with cluster size 2
    // Grid = num_sms total CTAs, cluster = {2,1,1} -> num_sms/2 clusters
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_sms);
    config.blockDim = dim3(TOTAL_THREADS);
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    config.attrs = attrs;
    config.numAttrs = 1;

    check_cuda_rt(cudaLaunchKernelEx(&config, fp8_gemm_kernel_2cta, (uint32_t) M, (uint32_t) N, (uint32_t) K, tma_a,
                      tma_b, tma_sfa, tma_sfb, tma_d, prob.sf, (uint32_t) num_sms),
        "kernel launch");

    // Check for async launch errors
    check_cuda_rt(cudaGetLastError(), "post-launch check");
}

// ============================================================================
// Test + Correctness Check
// ============================================================================

float fp8_e4m3_to_float(uint8_t val)
{
    uint32_t sign = (val >> 7) & 1;
    uint32_t exp = (val >> 3) & 0xF;
    uint32_t mant = val & 0x7;

    if (exp == 0 && mant == 0)
        return sign ? -0.0f : 0.0f;
    if (exp == 0xF && mant == 0x7)
        return sign ? -INFINITY : INFINITY;

    float result;
    if (exp == 0)
    {
        result = ldexpf((float) mant / 8.0f, -6);
    }
    else
    {
        result = ldexpf(1.0f + (float) mant / 8.0f, (int) exp - 7);
    }
    return sign ? -result : result;
}

float ue8m0_to_float(uint8_t val)
{
    return ldexpf(1.0f, (int) val - 127);
}

void cpu_fp8_gemm_reference(uint8_t const* A, uint8_t const* B, float* C, uint32_t const* sfa, uint32_t const* sfb,
    uint32_t M, uint32_t N, uint32_t K)
{
    uint32_t sfa_k = ceil_div(K, GRAN_K_A * 4);
    uint32_t sfb_k = ceil_div(K, GRAN_K_B * 4);

    for (uint32_t m = 0; m < M; m++)
    {
        for (uint32_t n = 0; n < N; n++)
        {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++)
            {
                float a_val = fp8_e4m3_to_float(A[m * K + k]);
                float b_val = fp8_e4m3_to_float(B[n * K + k]);

                uint32_t sf_k_group = k / (GRAN_K_A * 4);
                uint32_t sf_k_sub = (k / GRAN_K_A) % 4;

                uint32_t sfa_packed = sfa[sf_k_group * M + m];
                uint8_t sfa_byte = (sfa_packed >> (sf_k_sub * 8)) & 0xFF;
                float sfa_val = ue8m0_to_float(sfa_byte);

                uint32_t sfb_packed = sfb[sf_k_group * N + n];
                uint8_t sfb_byte = (sfb_packed >> (sf_k_sub * 8)) & 0xFF;
                float sfb_val = ue8m0_to_float(sfb_byte);

                acc += a_val * sfa_val * b_val * sfb_val;
            }
            C[m * N + n] = acc;
        }
    }
}

#ifndef FP8_GEMM_QUANT_NO_MAIN
int main(int argc, char** argv)
{
    // Initialize CUDA driver API FIRST — required before any driver API calls
    check_cuda(cuInit(0), "cuInit");

    uint32_t M = 4096, N = 4096, K = 4096;
    if (argc >= 4)
    {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("FP8 GEMM (2-CTA cluster): M=%u, N=%u, K=%u\n", M, N, K);
    printf("Tile: BLOCK_M=%u, BLOCK_N=%u, BLOCK_K=%u, LOAD_BLOCK_N=%u, STAGES=%u\n", BLOCK_M, BLOCK_N, BLOCK_K,
        LOAD_BLOCK_N, NUM_STAGES);
    printf("UMMA: M=%u, N=%u, K=%u (cta_group::2)\n", UMMA_M, UMMA_N, UMMA_K);
    printf("TMEM columns: %u (accum=%u, sfa=%u, sfb=%u)\n", NUM_TMEM_COLS, NUM_ACCUM_TMEM_COLS, NUM_SFA_TMEM_COLS,
        NUM_SFB_TMEM_COLS);

    // Print GPU info for debugging
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d, %d SMs, %zu bytes max SMEM)\n", prop.name, prop.major, prop.minor,
        prop.multiProcessorCount, prop.sharedMemPerBlockOptin);

    // Allocate
    __nv_fp8_e4m3 *d_A, *d_B;
    __nv_fp8_e4m3* d_D; // FP8 output
    uint8_t* d_sf;      // output UE8M0 scale byte [M, N/128]
    uint32_t *d_sfa, *d_sfb;

    uint32_t num_n_blocks = N / 128;
    // Packed UE8M0 A-scale buffer (deep_gemm layout): [num_packed_sf_k, m_aligned] int32.
    uint32_t num_packed_sf_k = (num_n_blocks + 3) / 4;
    uint32_t m_aligned = (M + 3) / 4 * 4;
    size_t size_A = (size_t) M * K;
    size_t size_B = (size_t) N * K;
    size_t size_D = (size_t) M * N * sizeof(__nv_fp8_e4m3);
    size_t size_sf = (size_t) num_packed_sf_k * m_aligned * sizeof(int32_t);
    uint32_t sfa_k = ceil_div(K, GRAN_K_A * 4);
    uint32_t sfb_k = ceil_div(K, GRAN_K_B * 4);
    size_t size_sfa = (size_t) sfa_k * M * sizeof(uint32_t);
    size_t size_sfb = (size_t) sfb_k * N * sizeof(uint32_t);

    check_cuda_rt(cudaMalloc(&d_A, size_A), "malloc A");
    check_cuda_rt(cudaMalloc(&d_B, size_B), "malloc B");
    check_cuda_rt(cudaMalloc(&d_D, size_D), "malloc D");
    check_cuda_rt(cudaMalloc(&d_sf, size_sf), "malloc sf");
    check_cuda_rt(cudaMalloc(&d_sfa, size_sfa), "malloc sfa");
    check_cuda_rt(cudaMalloc(&d_sfb, size_sfb), "malloc sfb");

    uint8_t* h_A = new uint8_t[size_A];
    uint8_t* h_B = new uint8_t[size_B];
    uint8_t* h_D = new uint8_t[(size_t) M * N]; // fp8 bytes
    uint8_t* h_sf = new uint8_t[size_sf];       // packed [num_packed_sf_k, m_aligned] int32 bytes
    uint32_t* h_sfa = new uint32_t[sfa_k * M];
    uint32_t* h_sfb = new uint32_t[sfb_k * N];
    float* h_ref = new float[M * N];

    srand(42);
    for (size_t i = 0; i < size_A; i++)
        h_A[i] = (7 << 3) | (rand() % 8);
    for (size_t i = 0; i < size_B; i++)
        h_B[i] = (7 << 3) | (rand() % 8);
    for (size_t i = 0; i < (size_t) sfa_k * M; i++)
        h_sfa[i] = 0x7F7F7F7F;
    for (size_t i = 0; i < (size_t) sfb_k * N; i++)
        h_sfb[i] = 0x7F7F7F7F;

    check_cuda_rt(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice), "copy A");
    check_cuda_rt(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice), "copy B");
    check_cuda_rt(cudaMemcpy(d_sfa, h_sfa, size_sfa, cudaMemcpyHostToDevice), "copy sfa");
    check_cuda_rt(cudaMemcpy(d_sfb, h_sfb, size_sfb, cudaMemcpyHostToDevice), "copy sfb");
    check_cuda_rt(cudaMemset(d_D, 0, size_D), "memset D");
    check_cuda_rt(cudaMemset(d_sf, 0, size_sf), "memset sf");

    GemmProblem prob;
    prob.M = M;
    prob.N = N;
    prob.K = K;
    prob.A = reinterpret_cast<__nv_fp8_e4m3*>(d_A);
    prob.B = reinterpret_cast<__nv_fp8_e4m3*>(d_B);
    prob.D = d_D;
    prob.sf = d_sf;
    prob.sfa = d_sfa;
    prob.sfb = d_sfb;

    printf("Warmup...\n");
    launch_fp8_gemm_2cta(prob);
    check_cuda_rt(cudaDeviceSynchronize(), "warmup sync");

    int num_iters = 20;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iters; i++)
    {
        launch_fp8_gemm_2cta(prob);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / num_iters;

    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    // Roofline SOL — derive peak from SM count + clock rate
    int dev;
    cudaGetDevice(&dev);
    int nsms;
    cudaDeviceGetAttribute(&nsms, cudaDevAttrMultiProcessorCount, dev);
    int gpu_clock_khz;
    cudaDeviceGetAttribute(&gpu_clock_khz, cudaDevAttrClockRate, dev);
    double gpu_clock_ghz = gpu_clock_khz / 1.0e6;

    // SM100a FP8 dense: 4500 TFLOPS @ 160 SMs @ 2.1 GHz → 13.39 TFLOPS/SM/GHz
    constexpr double FP8_TFLOPS_PER_SM_PER_GHZ = 4500.0 / 160.0 / 2.1;
    double compute_roof = FP8_TFLOPS_PER_SM_PER_GHZ * gpu_clock_ghz * nsms;

    int mem_clock_khz, mem_bus_width;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth, dev);
    double hbm_bw_gbps = (double) mem_clock_khz * 1e3 * mem_bus_width / 8.0 * 2.0 / 1e9;

    double bytes_ab = (double) M * K + (double) N * K;
    double bytes_d = (double) M * N * 2;
    uint32_t sfa_k_dim = ceil_div(K, GRAN_K_A * 4);
    uint32_t sfb_k_dim = ceil_div(K, GRAN_K_B * 4);
    double bytes_sf = (double) sfa_k_dim * M * 4 + (double) sfb_k_dim * N * 4;
    double total_bytes = bytes_ab + bytes_d + bytes_sf;
    double ai = flops / total_bytes;

    double memory_roof = hbm_bw_gbps * ai / 1e3;
    double roofline = (compute_roof < memory_roof) ? compute_roof : memory_roof;
    double sol = tflops / roofline * 100.0;

    printf("\n===== Roofline Analysis =====\n");
    printf("GPU: %d SMs @ %.0f MHz (%.2f GHz)\n", nsms, gpu_clock_khz / 1e3, gpu_clock_ghz);
    printf("Arithmetic intensity: %.1f FLOP/byte\n", ai);
    printf("Compute roof: %.1f TFLOPS (FP8 dense, %.1f TFLOPS/SM/GHz × %d SMs × %.2f GHz)\n", compute_roof,
        FP8_TFLOPS_PER_SM_PER_GHZ, nsms, gpu_clock_ghz);
    printf("Memory roof:  %.1f TFLOPS (HBM BW=%.0f GB/s, AI=%.1f)\n", memory_roof, hbm_bw_gbps, ai);
    printf("Roofline ceiling: %.1f TFLOPS (%s)\n", roofline,
        compute_roof < memory_roof ? "compute-bound" : "memory-bound");

    printf("\n===== Performance =====\n");
    printf("Average time: %.3f ms\n", avg_ms);
    printf("Throughput: %.1f TFLOPS (FP8)\n", tflops);
    printf("SOL (vs roofline): %.1f%%\n", sol);

    // Correctness check: dequant(GPU fp8 D × per-block scale) vs fp32 GEMM, AND vs the
    // reference path "fp32 GEMM → per-128-block fp8 quant → dequant" (isolates the GEMM
    // error from the quant error).
    if (M <= 512 && N <= 512 && K <= 512)
    {
        printf("\nRunning CPU reference (GEMM + per-128-block fp8 quant) for correctness...\n");
        cpu_fp8_gemm_reference(h_A, h_B, h_ref, h_sfa, h_sfb, M, N, K);

        check_cuda_rt(cudaMemcpy(h_D, d_D, size_D, cudaMemcpyDeviceToHost), "copy D back");
        check_cuda_rt(cudaMemcpy(h_sf, d_sf, size_sf, cudaMemcpyDeviceToHost), "copy sf back");

        // Reference quant: per (row, 128-block) amax over the fp32 GEMM, quantize→e4m3→dequant.
        float max_rel_vs_gemm = 0, max_rel_vs_refquant = 0;
        for (uint32_t m = 0; m < M; m++)
        {
            for (uint32_t nb = 0; nb < num_n_blocks; nb++)
            {
                float amax = 0;
                for (uint32_t j = 0; j < 128; j++)
                    amax = fmaxf(amax, fabsf(h_ref[m * N + nb * 128 + j]));
                // Reference UE8M0 scale: smallest 2^e >= amax/448 (same ceil-to-pow2 as the kernel).
                float rscale, rinv;
                if (amax > 0)
                {
                    float s = amax / 448.0f;
                    int ef;
                    float mant = frexpf(s, &ef);
                    int e = (mant > 0.5f) ? ef : (ef - 1);
                    rscale = ldexpf(1.0f, e);
                    rinv = ldexpf(1.0f, -e);
                }
                else
                {
                    rscale = 0.0f;
                    rinv = 0.0f;
                }
                for (uint32_t j = 0; j < 128; j++)
                {
                    uint32_t idx = m * N + nb * 128 + j;
                    float gemm = h_ref[idx];
                    float ref_deq = (float) (__nv_fp8_e4m3) (gemm * rinv) * rscale; // ref path
                    // Unpack UE8M0 byte from the packed [num_packed_sf_k, m_aligned] int32 layout.
                    uint32_t packed = reinterpret_cast<uint32_t*>(h_sf)[(size_t) (nb >> 2) * m_aligned + m];
                    uint8_t sf_b = (packed >> ((nb & 3) * 8)) & 0xFF;
                    float gpu_deq = (float) (*reinterpret_cast<__nv_fp8_e4m3*>(&h_D[idx])) * ue8m0_to_float(sf_b);
                    float rg = (gemm != 0) ? fabsf(gpu_deq - gemm) / fabsf(gemm) : fabsf(gpu_deq - gemm);
                    float rr = (ref_deq != 0) ? fabsf(gpu_deq - ref_deq) / fabsf(ref_deq) : fabsf(gpu_deq - ref_deq);
                    max_rel_vs_gemm = fmaxf(max_rel_vs_gemm, rg);
                    max_rel_vs_refquant = fmaxf(max_rel_vs_refquant, rr);
                }
            }
        }
        printf("Max rel err vs fp32 GEMM (incl. fp8 quant):   %e\n", max_rel_vs_gemm);
        printf("Max rel err vs reference per-block quant:     %e\n", max_rel_vs_refquant);
        printf("Correctness: %s\n", max_rel_vs_refquant < 0.02f ? "PASS" : "FAIL");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    cudaFree(d_sf);
    cudaFree(d_sfa);
    cudaFree(d_sfb);
    delete[] h_A;
    delete[] h_B;
    delete[] h_D;
    delete[] h_sf;
    delete[] h_sfa;
    delete[] h_sfb;
    delete[] h_ref;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
#endif // FP8_GEMM_QUANT_NO_MAIN
