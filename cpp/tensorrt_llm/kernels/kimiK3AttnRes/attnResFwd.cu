/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Fused Kimi K3 attention-residual forward for Blackwell (sm_100 family).
//
// Warp-specialized online softmax + residual + RMSNorm:
//   - 1 producer warp issues cp.async.bulk row loads into shared memory.
//   - 8 consumer warps compute reductions and output.
//   - Q=res_weight*rms_weight remains in registers across persistent tokens.
//   - V rows are converted once and cached as FP32 in TMEM between passes.
//
// Contract: B=1, N<=12, H in [4096,8192] and divisible by 1024; checked at
// the Torch-op bridge. The kernel uses separate layer/block residual inputs
// and does not require a concatenated V tensor.
//
// Source-integrated from the NVIDIA+Moonshot jointly developed
// Attention_residual kernel at e7f934124acc915575f9f7561f9d1e373ab43089.

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.h"

#include <algorithm>
#include <cfloat>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <mutex>

namespace
{

using bf16_t = __nv_bfloat16;

static constexpr int ATTN_RES_BLOCK = 256;
static constexpr int ATTN_RES_WARPS = ATTN_RES_BLOCK / 32;

__inline__ __device__ float warp_reduce_sum(float val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_sum(float val, float* ws)
{
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0)
        ws[wid] = val;
    __syncthreads();
    val = (threadIdx.x < ATTN_RES_WARPS) ? ws[threadIdx.x] : 0.f;
    if (wid == 0)
        val = warp_reduce_sum(val);
    return val;
}

__device__ __forceinline__ bf16_t const* v_addr(
    bf16_t const* block_res, bf16_t const* layer_res, int n, int N, int t, int b, int T, int B, int H)
{
    if (n < N - 1)
        return block_res + (((long long) n * T + t) * B + b) * H;
    return layer_res + ((long long) t * B + b) * H;
}

namespace sm100
{

CUTE_DEVICE
void tcgen05_after_thread_sync()
{
    asm volatile("tcgen05.fence::after_thread_sync;");
}

CUTE_DEVICE
void umma_arrive_noelect(uint64_t& bar_ptr)
{
    uint64_t bar_addr = cute::cast_smem_ptr_to_uint(&bar_ptr);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" : : "l"(bar_addr));
}

CUTE_DEVICE
float2 float2_sub(float2 const& a, float2 const& b)
{
    float2 c;
    asm volatile("sub.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

CUTE_DEVICE
float2 float2_mul(float2 const& a, float2 const& b)
{
    float2 c;
    asm volatile("mul.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

CUTE_DEVICE
float2 float2_fma(float2 const& a, float2 const& b, float2 const& c)
{
    float2 d;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(d))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

CUTE_DEVICE
float2 float2_add(float2 const& a, float2 const& b)
{
    float2 c;
    asm volatile("add.rn.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

template <int N, typename T>
CUTE_DEVICE void tmem_ld_32dp32bNx(uint32_t const& src_addr, T* dst_ptr_)
{
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst_ptr_);
    if constexpr (N == 8)
    {
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32"
            "{%0, %1, %2, %3, %4, %5, %6, %7},"
            "[%8];\n"
            : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]), "=r"(dst_ptr[3]), "=r"(dst_ptr[4]),
            "=r"(dst_ptr[5]), "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
            : "r"(src_addr));
    }
    else
    {
        static_assert(N == 4, "attn_res TMEM helpers support x4 and x8");
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x4.b32"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]), "=r"(dst_ptr[3])
            : "r"(src_addr));
    }
}

template <int N, typename T>
CUTE_DEVICE void tmem_st_32dp32bNx(uint32_t const& dst_addr, T* src_ptr_)
{
    uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src_ptr_);
    if constexpr (N == 8)
    {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x8.b32"
            "[%8], {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            :
            : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
            "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(dst_addr));
    }
    else
    {
        static_assert(N == 4, "attn_res TMEM helpers support x4 and x8");
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x4.b32"
            "[%4], {%0, %1, %2, %3};\n"
            :
            : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(dst_addr));
    }
}

namespace fwd_prod_v2
{

using namespace cute;

constexpr int K_TILE = 1024;
constexpr int N_MAX = 12;
constexpr int N_CHUNK_DEFAULT = 4;
constexpr int CHUNK_DEPTH = 2;
constexpr int BLK = 288;                   // 1 producer warp + 8 consumer warps
constexpr int CONSUMER_THREADS = BLK - 32; // 256
constexpr int CONSUMER_WARPS = CONSUMER_THREADS / 32;
constexpr int CONSUMER_GROUPS = 2;         // two 128-thread consumer groups
constexpr int CONSUMER_THREADS_PER_GROUP = CONSUMER_THREADS / CONSUMER_GROUPS;
constexpr int TMEM_Q_COLS_PER_GROUP = 32;
constexpr int TMEM_Q_COLS_TOTAL = 2 * TMEM_Q_COLS_PER_GROUP;

template <int NC>
struct FwdSmemPlan
{
    alignas(16) uint64_t bar_ready[CHUNK_DEPTH];
    alignas(16) uint64_t bar_consumed[CHUNK_DEPTH];
    alignas(16) float2 ws_stats[CONSUMER_WARPS][NC];
    alignas(16) float logits_all[N_MAX];
    uint32_t tmem_base;
};

__device__ __forceinline__ void cp_async_bulk(void* smem_dst, void const* gmem_src, int bytes, uint64_t& mbar)
{
    uint32_t s = cute::cast_smem_ptr_to_uint(smem_dst);
    uint32_t m = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(s),
                 "l"(gmem_src), "r"(bytes), "r"(m)
                 : "memory");
}

template <int H, int NC = N_CHUNK_DEFAULT, bool RELEASE_TMEM = false, bool FULL_N12 = false>
__global__ void __launch_bounds__(BLK, 1) attn_res_fwd_online_v2_kernel(bf16_t const* __restrict__ block_res,
    bf16_t const* __restrict__ layer_res, bf16_t const* __restrict__ res_w, bf16_t const* __restrict__ rms_w,
    bf16_t* __restrict__ output, float* __restrict__ rsigma_out, float* __restrict__ probs_out,
    float* __restrict__ logits_out, int N, int T, int B, float rms_eps)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    constexpr float LOG2_E = 1.4426950408889634f;
    constexpr int N_CHUNK = NC;
    constexpr int NUM_BUFS = CHUNK_DEPTH * NC;
    constexpr int NHT = H / K_TILE;
    constexpr int SLICES_PER_GROUP = (NHT + CONSUMER_GROUPS - 1) / CONSUMER_GROUPS;
    constexpr int VEC = 8;
    constexpr int ACC_PER_THREAD = H == 7168 ? 28 : SLICES_PER_GROUP * VEC;
    constexpr int TMEM_V_COLS_PER_GROUP = SLICES_PER_GROUP * N_CHUNK * VEC;
    constexpr int TMEM_COLS_TOTAL = CONSUMER_GROUPS * TMEM_V_COLS_PER_GROUP;
    constexpr int TMEM_COLS_ALLOC = 256;
    static_assert(TMEM_COLS_TOTAL <= TMEM_COLS_ALLOC);
    static_assert(H >= 4096 && H <= 8192);
    static_assert(H % K_TILE == 0);

    int const tid = threadIdx.x;
    int const wid = tid >> 5;
    int const lane = tid & 31;
    int const TB = FULL_N12 ? 1024 : T;
    int const num_ctas = gridDim.x;
    int const num_chunks = (N + N_CHUNK - 1) / N_CHUNK;

    int const comp_wid = wid - 1;
    int const comp_tid = tid - 32;
    int const group = (comp_wid >= 4) ? 1 : 0;
    int const ct_in_group = (comp_tid >= 0) ? (comp_tid & (CONSUMER_THREADS_PER_GROUP - 1)) : -1;
    int const k_local = ct_in_group * VEC;

    extern __shared__ char smem_raw[];
    bf16_t* v_bufs = reinterpret_cast<bf16_t*>(smem_raw); // [NUM_BUFS][H]
    constexpr size_t V_BYTES = (size_t) NUM_BUFS * H * sizeof(bf16_t);
    FwdSmemPlan<NC>& plan = *reinterpret_cast<FwdSmemPlan<NC>*>(smem_raw + V_BYTES);

    auto slot_of = [](long long gci, int n) { return (int) (gci % CHUNK_DEPTH) * N_CHUNK + n; };
    auto phase_of = [](long long gci) { return (int) ((gci / CHUNK_DEPTH) & 1); };
    auto buf_ptr = [&](int slot) -> bf16_t* { return v_bufs + slot * H; };

    if (wid == 0 && elect_one_sync())
    {
#pragma unroll
        for (int i = 0; i < CHUNK_DEPTH; i++)
        {
            cute::initialize_barrier(plan.bar_ready[i], 1);
            cute::initialize_barrier(plan.bar_consumed[i], CONSUMER_WARPS);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (wid == 1)
    {
        cute::TMEM::Allocator1Sm alloc;
        alloc.allocate(TMEM_COLS_ALLOC, &plan.tmem_base);
        if constexpr (RELEASE_TMEM)
        {
            alloc.release_allocation_lock();
        }
    }
    __syncthreads();

    const uint32_t my_v_tmem = (comp_tid >= 0) ? (plan.tmem_base + group * TMEM_V_COLS_PER_GROUP) : 0;

    float q_cache[ACC_PER_THREAD];
    if (comp_tid >= 0)
    {
#pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++)
        {
            if constexpr (H == 7168)
            {
                if (si == SLICES_PER_GROUP - 1)
                {
                    int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
#pragma unroll
                    for (int j = 0; j < 4; j++)
                    {
                        int h = h_base + j;
                        q_cache[si * VEC + j] = __bfloat162float(rms_w[h]) * __bfloat162float(res_w[h]);
                    }
                    continue;
                }
            }
            int dt = si * CONSUMER_GROUPS + group;
            if (dt >= NHT)
                continue;
            int h_base = dt * K_TILE + k_local;
#pragma unroll
            for (int j = 0; j < VEC; j++)
            {
                int h = h_base + j;
                q_cache[si * VEC + j] = __bfloat162float(rms_w[h]) * __bfloat162float(res_w[h]);
            }
        }
    }

    if (wid == 0)
    {
        if (elect_one_sync())
        {
            long long gci = 0;
            for (int tb = blockIdx.x; tb < TB; tb += num_ctas)
            {
                for (int ci = 0; ci < num_chunks; ci++, gci++)
                {
                    int ns = ci * N_CHUNK;
                    int an = FULL_N12 ? N_CHUNK : min(N_CHUNK, N - ns);
                    int chunk_slot = (int) (gci % CHUNK_DEPTH);
                    int pc = phase_of(gci);
                    cute::wait_barrier(plan.bar_consumed[chunk_slot], pc ^ 1);
                    cute::set_barrier_transaction_bytes(plan.bar_ready[chunk_slot], an * H * (int) sizeof(bf16_t));
#pragma unroll
                    for (int n = 0; n < N_CHUNK; n++)
                    {
                        if constexpr (!FULL_N12)
                        {
                            if (n >= an)
                                continue;
                        }
                        int slot = slot_of(gci, n);
                        int const ng = ns + n;
                        bf16_t const* src = (ng < (FULL_N12 ? 11 : N - 1)) ? block_res + ((long long) ng * T + tb) * H
                                                                           : layer_res + (long long) tb * H;
                        cp_async_bulk(buf_ptr(slot), src, H * sizeof(bf16_t), plan.bar_ready[chunk_slot]);
                    }
                }
            }
        }
    }
    else
    {
        float acc32[ACC_PER_THREAD] = {};
        float eps_cache;
        asm volatile("mov.b32 %0, %1;" : "=f"(eps_cache) : "f"(rms_eps));

        long long gci = 0;
        for (int tb = blockIdx.x; tb < TB; tb += num_ctas)
        {
            float m_running = -FLT_MAX;
            float s_running = 0.f;
#pragma unroll
            for (int i = 0; i < ACC_PER_THREAD; i++)
            {
                acc32[i] = 0.f;
            }

            for (int ci = 0; ci < num_chunks; ci++, gci++)
            {
                int ns = ci * N_CHUNK;
                int an = FULL_N12 ? N_CHUNK : min(N_CHUNK, N - ns);
                int chunk_slot = (int) (gci % CHUNK_DEPTH);
                int pr = phase_of(gci);
                float2 sq_local[N_CHUNK] = {};
                float2 dot_local[N_CHUNK] = {};
                cute::wait_barrier(plan.bar_ready[chunk_slot], pr);

                auto pass_A_body = [&](auto AN_TOK)
                {
                    constexpr int AN = decltype(AN_TOK)::value;
#pragma unroll
                    for (int si = 0; si < SLICES_PER_GROUP; si++)
                    {
                        if constexpr (H == 7168)
                        {
                            if (si == SLICES_PER_GROUP - 1)
                            {
                                int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
                                const float* qv = &q_cache[si * VEC];
#pragma unroll
                                for (int n = 0; n < AN; n++)
                                {
                                    int slot = slot_of(gci, n);
                                    int2 vp = *reinterpret_cast<const int2*>(buf_ptr(slot) + h_base);
                                    __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
                                    float2 f[2] = {__bfloat1622float2(v2[0]), __bfloat1622float2(v2[1])};
                                    if constexpr (FULL_N12)
                                    {
                                        if (n == AN - 1 && lane == 0)
                                        {
                                            cute::arrive_barrier(plan.bar_consumed[chunk_slot]);
                                        }
                                    }
                                    tmem_st_32dp32bNx<4>(
                                        my_v_tmem + (si * N_CHUNK + n) * VEC, reinterpret_cast<float*>(f));
                                    sq_local[n] = float2_fma(f[0], f[0], sq_local[n]);
                                    sq_local[n] = float2_fma(f[1], f[1], sq_local[n]);
                                    dot_local[n] = float2_fma(f[0], make_float2(qv[0], qv[1]), dot_local[n]);
                                    dot_local[n] = float2_fma(f[1], make_float2(qv[2], qv[3]), dot_local[n]);
                                }
                                continue;
                            }
                        }
                        int dt = si * CONSUMER_GROUPS + group;
                        if (dt >= NHT)
                            continue;
                        const float* qv = &q_cache[si * VEC];

#pragma unroll
                        for (int n = 0; n < AN; n++)
                        {
                            int slot = slot_of(gci, n);
                            int4 vp = *reinterpret_cast<const int4*>(buf_ptr(slot) + dt * K_TILE + k_local);
                            __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
                            float2 f[4] = {__bfloat1622float2(v2[0]), __bfloat1622float2(v2[1]),
                                __bfloat1622float2(v2[2]), __bfloat1622float2(v2[3])};
                            tmem_st_32dp32bNx<VEC>(my_v_tmem + (si * N_CHUNK + n) * VEC, reinterpret_cast<float*>(f));
                            sq_local[n] = float2_fma(f[0], f[0], sq_local[n]);
                            sq_local[n] = float2_fma(f[1], f[1], sq_local[n]);
                            sq_local[n] = float2_fma(f[2], f[2], sq_local[n]);
                            sq_local[n] = float2_fma(f[3], f[3], sq_local[n]);
                            dot_local[n] = float2_fma(f[0], make_float2(qv[0], qv[1]), dot_local[n]);
                            dot_local[n] = float2_fma(f[1], make_float2(qv[2], qv[3]), dot_local[n]);
                            dot_local[n] = float2_fma(f[2], make_float2(qv[4], qv[5]), dot_local[n]);
                            dot_local[n] = float2_fma(f[3], make_float2(qv[6], qv[7]), dot_local[n]);
                        }
                    }
                    if constexpr (!FULL_N12)
                    {
                        cutlass::arch::fence_view_async_tmem_store();
                    }
                };
                if constexpr (FULL_N12)
                {
                    pass_A_body(std::integral_constant<int, N_CHUNK>{});
                }
                else if constexpr (NC == 4)
                {
                    switch (an)
                    {
                    case 4: pass_A_body(std::integral_constant<int, 4>{}); break;
                    case 3: pass_A_body(std::integral_constant<int, 3>{}); break;
                    case 2: pass_A_body(std::integral_constant<int, 2>{}); break;
                    case 1: pass_A_body(std::integral_constant<int, 1>{}); break;
                    default: __builtin_unreachable();
                    }
                }
                else if constexpr (NC == 3)
                {
                    switch (an)
                    {
                    case 3: pass_A_body(std::integral_constant<int, 3>{}); break;
                    case 2: pass_A_body(std::integral_constant<int, 2>{}); break;
                    case 1: pass_A_body(std::integral_constant<int, 1>{}); break;
                    default: __builtin_unreachable();
                    }
                }
                else
                {
                    static_assert(NC == 2);
                    switch (an)
                    {
                    case 2: pass_A_body(std::integral_constant<int, 2>{}); break;
                    case 1: pass_A_body(std::integral_constant<int, 1>{}); break;
                    default: __builtin_unreachable();
                    }
                }
                if constexpr (!FULL_N12)
                {
                    if (lane == 0)
                    {
                        cute::arrive_barrier(plan.bar_consumed[chunk_slot]);
                    }
                }

                float2 reduce_pair[N_CHUNK];
#pragma unroll
                for (int n = 0; n < N_CHUNK; n++)
                {
                    reduce_pair[n] = make_float2(sq_local[n].x + sq_local[n].y, dot_local[n].x + dot_local[n].y);
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                {
#pragma unroll
                    for (int n = 0; n < N_CHUNK; n++)
                    {
                        uint64_t packed = reinterpret_cast<uint64_t&>(reduce_pair[n]);
                        packed = __shfl_xor_sync(0xffffffff, packed, offset);
                        float2 other = reinterpret_cast<float2&>(packed);
                        reduce_pair[n] = float2_add(reduce_pair[n], other);
                    }
                }
                if constexpr (FULL_N12)
                {
                    cutlass::arch::fence_view_async_tmem_store();
                }
                if (lane == 0)
                {
#pragma unroll
                    for (int n = 0; n < N_CHUNK; n++)
                    {
                        plan.ws_stats[comp_wid][n] = reduce_pair[n];
                    }
                }
                cutlass::arch::NamedBarrier::sync(CONSUMER_THREADS, 0);

                float local_rsig = 0.f;
                float local_logit = 0.f;
                auto cross_warp_tail = [&](int n)
                {
                    float2 totals = {};
#pragma unroll
                    for (int w = 0; w < CONSUMER_WARPS; w++)
                    {
                        totals = float2_add(totals, plan.ws_stats[w][n]);
                    }
                    local_rsig = rsqrtf(totals.x / H + eps_cache);
                    local_logit = totals.y * local_rsig;
                };
                if constexpr (FULL_N12)
                {
                    cross_warp_tail(lane & (N_CHUNK - 1));
                }
                else if (lane < N_CHUNK)
                {
                    cross_warp_tail(lane);
                }
                float logit_n[N_CHUNK];
#pragma unroll
                for (int n = 0; n < N_CHUNK; n++)
                {
                    logit_n[n] = __shfl_sync(0xffffffff, local_logit, n);
                }

                float m_chunk = -FLT_MAX;
                if constexpr (FULL_N12)
                {
                    float m01 = fmaxf(logit_n[0], logit_n[1]);
                    float m23 = fmaxf(logit_n[2], logit_n[3]);
                    m_chunk = fmaxf(m01, m23);
                }
                else
                {
#pragma unroll
                    for (int n = 0; n < N_CHUNK; n++)
                    {
                        if (n < an)
                        {
                            m_chunk = fmaxf(m_chunk, logit_n[n]);
                        }
                    }
                }
                float m_new = fmaxf(m_running, m_chunk);
                float corr = exp2f((m_running - m_new) * LOG2_E);
                float w_n[N_CHUNK] = {};
                float w_sum = 0.f;
                if constexpr (FULL_N12)
                {
#pragma unroll
                    for (int n = 0; n < N_CHUNK; n++)
                    {
                        w_n[n] = exp2f((logit_n[n] - m_new) * LOG2_E);
                    }
                    w_sum = (w_n[0] + w_n[1]) + (w_n[2] + w_n[3]);
                }
                else
                {
#pragma unroll
                    for (int n = 0; n < N_CHUNK; n++)
                    {
                        if (n < an)
                        {
                            w_n[n] = exp2f((logit_n[n] - m_new) * LOG2_E);
                            w_sum += w_n[n];
                        }
                    }
                }

                auto pass_B_body = [&](auto AN_TOK)
                {
                    constexpr int AN = decltype(AN_TOK)::value;
#pragma unroll
                    for (int si = 0; si < SLICES_PER_GROUP; si++)
                    {
                        if constexpr (H == 7168)
                        {
                            if (si == SLICES_PER_GROUP - 1)
                            {
                                float2 corr2 = make_float2(corr, corr);
                                float2 a[2];
#pragma unroll
                                for (int j = 0; j < 2; j++)
                                {
                                    float2 old = make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
                                    a[j] = float2_mul(old, corr2);
                                }
                                float2 f_cache[AN][2];
#pragma unroll
                                for (int n = 0; n < AN; n++)
                                {
                                    tmem_ld_32dp32bNx<4>(
                                        my_v_tmem + (si * N_CHUNK + n) * VEC, reinterpret_cast<float*>(f_cache[n]));
                                }
#pragma unroll
                                for (int n = 0; n < AN; n++)
                                {
                                    float2 wn = make_float2(w_n[n], w_n[n]);
#pragma unroll
                                    for (int j = 0; j < 2; j++)
                                    {
                                        a[j] = float2_fma(wn, f_cache[n][j], a[j]);
                                    }
                                }
#pragma unroll
                                for (int j = 0; j < 2; j++)
                                {
                                    acc32[si * VEC + 2 * j] = a[j].x;
                                    acc32[si * VEC + 2 * j + 1] = a[j].y;
                                }
                                continue;
                            }
                        }
                        int dt = si * CONSUMER_GROUPS + group;
                        if (dt >= NHT)
                            continue;
                        float2 a[VEC / 2];
                        float2 corr2 = make_float2(corr, corr);
#pragma unroll
                        for (int j = 0; j < VEC / 2; j++)
                        {
                            float2 old = make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
                            a[j] = float2_mul(old, corr2);
                        }
                        float2 f_cache[AN][VEC / 2];
#pragma unroll
                        for (int n = 0; n < AN; n++)
                        {
                            tmem_ld_32dp32bNx<VEC>(
                                my_v_tmem + (si * N_CHUNK + n) * VEC, reinterpret_cast<float*>(f_cache[n]));
                        }
#pragma unroll
                        for (int n = 0; n < AN; n++)
                        {
                            float2 wn = make_float2(w_n[n], w_n[n]);
#pragma unroll
                            for (int j = 0; j < VEC / 2; j++)
                            {
                                a[j] = float2_fma(wn, f_cache[n][j], a[j]);
                            }
                        }
#pragma unroll
                        for (int j = 0; j < VEC / 2; j++)
                        {
                            acc32[si * VEC + 2 * j] = a[j].x;
                            acc32[si * VEC + 2 * j + 1] = a[j].y;
                        }
                    }
                };
                if constexpr (FULL_N12)
                {
                    pass_B_body(std::integral_constant<int, N_CHUNK>{});
                }
                else if constexpr (NC == 4)
                {
                    switch (an)
                    {
                    case 4: pass_B_body(std::integral_constant<int, 4>{}); break;
                    case 3: pass_B_body(std::integral_constant<int, 3>{}); break;
                    case 2: pass_B_body(std::integral_constant<int, 2>{}); break;
                    case 1: pass_B_body(std::integral_constant<int, 1>{}); break;
                    default: __builtin_unreachable();
                    }
                }
                else if constexpr (NC == 3)
                {
                    switch (an)
                    {
                    case 3: pass_B_body(std::integral_constant<int, 3>{}); break;
                    case 2: pass_B_body(std::integral_constant<int, 2>{}); break;
                    case 1: pass_B_body(std::integral_constant<int, 1>{}); break;
                    default: __builtin_unreachable();
                    }
                }
                else
                {
                    static_assert(NC == 2);
                    switch (an)
                    {
                    case 2: pass_B_body(std::integral_constant<int, 2>{}); break;
                    case 1: pass_B_body(std::integral_constant<int, 1>{}); break;
                    default: __builtin_unreachable();
                    }
                }

                s_running = s_running * corr + w_sum;
                m_running = m_new;

                if (comp_wid == 0 && lane < an)
                {
                    int ng = ns + lane;
                    rsigma_out[(long long) ng * TB + tb] = local_rsig;
                    plan.logits_all[ng] = local_logit;
                }
            }
            // Publish the final chunk's plan.logits_all stores before the
            // cross-lane reads in consumer warp 0 below (earlier chunks are
            // covered by the NamedBarrier inside the loop).
            __syncwarp();

            float inv_s = 1.f / s_running;
            bf16_t* out_ptr = output + (long long) tb * H;
#pragma unroll
            for (int si = 0; si < SLICES_PER_GROUP; si++)
            {
                if constexpr (H == 7168)
                {
                    if (si == SLICES_PER_GROUP - 1)
                    {
                        int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
                        uint2 ov;
                        __nv_bfloat162* ov2 = reinterpret_cast<__nv_bfloat162*>(&ov);
                        float2 inv2 = make_float2(inv_s, inv_s);
#pragma unroll
                        for (int j = 0; j < 2; j++)
                        {
                            float2 old = make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
                            ov2[j] = __float22bfloat162_rn(float2_mul(old, inv2));
                        }
                        *reinterpret_cast<uint2*>(out_ptr + h_base) = ov;
                        continue;
                    }
                }
                int dt = si * CONSUMER_GROUPS + group;
                if (dt >= NHT)
                    continue;
                int h_base = dt * K_TILE + k_local;
                uint4 ov;
                __nv_bfloat162* ov2 = reinterpret_cast<__nv_bfloat162*>(&ov);
                float2 inv2 = make_float2(inv_s, inv_s);
#pragma unroll
                for (int j = 0; j < VEC / 2; j++)
                {
                    float2 old = make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
                    ov2[j] = __float22bfloat162_rn(float2_mul(old, inv2));
                }
                *reinterpret_cast<uint4*>(out_ptr + h_base) = ov;
            }

            if (comp_wid == 0 && lane < (FULL_N12 ? 12 : N))
            {
                long long out_idx = (long long) lane * TB + tb;
                float lg = plan.logits_all[lane];
                logits_out[out_idx] = lg;
                probs_out[out_idx] = exp2f((lg - m_running) * LOG2_E) * inv_s;
            }
        }
    }

    if (wid > 0)
    {
        cutlass::arch::NamedBarrier::sync(CONSUMER_THREADS, 2);
    }
    if (wid == 1)
    {
        cute::TMEM::Allocator1Sm alloc;
        alloc.free(plan.tmem_base, TMEM_COLS_ALLOC);
    }
#else
    if (cute::thread0())
        printf("attn_res_fwd_online_v2_kernel requires sm_100a\n");
#endif
}

// N=1 specialization: softmax is degenerate, so output is the layer row.
// Tile multiple contiguous TB rows per CTA to reduce cp.async.bulk overhead.
template <int H, int TB_TILE, bool RELEASE_TMEM = true>
__global__ void __launch_bounds__(BLK, 1)
    attn_res_fwd_n1_ttile_kernel(bf16_t const* __restrict__ layer_res, bf16_t const* __restrict__ res_w,
        bf16_t const* __restrict__ rms_w, bf16_t* __restrict__ output, float* __restrict__ rsigma_out,
        float* __restrict__ probs_out, float* __restrict__ logits_out, int T, int B, float rms_eps)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    constexpr int NHT = H / K_TILE;
    constexpr int SLICES_PER_GROUP = (NHT + CONSUMER_GROUPS - 1) / CONSUMER_GROUPS;
    constexpr int VEC = 8;
    constexpr int ACC_PER_THREAD = SLICES_PER_GROUP * VEC;
    static_assert(H == 4096 || H == 8192);

    int const tid = threadIdx.x;
    int const wid = tid >> 5;
    int const lane = tid & 31;
    int const TB = T * B;
    int const comp_wid = wid - 1;
    int const comp_tid = tid - 32;
    int const group = (comp_wid >= 4) ? 1 : 0;
    int const ct_in_group = (comp_tid >= 0) ? (comp_tid & (CONSUMER_THREADS_PER_GROUP - 1)) : -1;
    int const k_local = ct_in_group * VEC;

    extern __shared__ char smem_raw[];
    bf16_t* v_tiles = reinterpret_cast<bf16_t*>(smem_raw);
    constexpr size_t V_BYTES = (size_t) CHUNK_DEPTH * TB_TILE * H * sizeof(bf16_t);
    FwdSmemPlan<1>& plan = *reinterpret_cast<FwdSmemPlan<1>*>(smem_raw + V_BYTES);

    auto phase_of = [](long long tile_i) { return (int) ((tile_i / CHUNK_DEPTH) & 1); };
    auto tile_ptr = [&](int slot, int row) -> bf16_t* { return v_tiles + ((slot * TB_TILE + row) * H); };

    if (wid == 0 && elect_one_sync())
    {
#pragma unroll
        for (int i = 0; i < CHUNK_DEPTH; i++)
        {
            cute::initialize_barrier(plan.bar_ready[i], 1);
            cute::initialize_barrier(plan.bar_consumed[i], CONSUMER_THREADS);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (wid == 1)
    {
        cute::TMEM::Allocator1Sm alloc;
        alloc.allocate(TMEM_Q_COLS_TOTAL, &plan.tmem_base);
        if constexpr (RELEASE_TMEM)
        {
            alloc.release_allocation_lock();
        }
    }
    __syncthreads();

    const uint32_t my_tmem = (comp_tid >= 0) ? (plan.tmem_base + ((comp_wid >= 4) ? TMEM_Q_COLS_PER_GROUP : 0)) : 0;

    if (comp_tid >= 0)
    {
        float q32[ACC_PER_THREAD];
#pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++)
        {
            int dt = si * CONSUMER_GROUPS + group;
            if (dt >= NHT)
                continue;
            int h_base = dt * K_TILE + k_local;
#pragma unroll
            for (int j = 0; j < VEC; j++)
            {
                int h = h_base + j;
                q32[si * VEC + j] = __bfloat162float(rms_w[h]) * __bfloat162float(res_w[h]);
            }
        }
#pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++)
        {
            int dt = si * CONSUMER_GROUPS + group;
            if (dt >= NHT)
                continue;
            tmem_st_32dp32bNx<VEC>(my_tmem + si * VEC, &q32[si * VEC]);
        }
        cutlass::arch::fence_view_async_tmem_store();
    }
    __syncthreads();

    if (wid == 0)
    {
        if (elect_one_sync())
        {
            long long tile_i = 0;
            for (int tb0 = blockIdx.x * TB_TILE; tb0 < TB; tb0 += gridDim.x * TB_TILE, tile_i++)
            {
                int rows = min(TB_TILE, TB - tb0);
                int slot = (int) (tile_i % CHUNK_DEPTH);
                int pc = phase_of(tile_i);
                cute::wait_barrier(plan.bar_consumed[slot], pc ^ 1);
                cute::set_barrier_transaction_bytes(plan.bar_ready[slot], rows * H * (int) sizeof(bf16_t));
                cp_async_bulk(tile_ptr(slot, 0), layer_res + (long long) tb0 * H, rows * H * sizeof(bf16_t),
                    plan.bar_ready[slot]);
            }
        }
    }
    else
    {
        long long tile_i = 0;
        for (int tb0 = blockIdx.x * TB_TILE; tb0 < TB; tb0 += gridDim.x * TB_TILE, tile_i++)
        {
            int rows = min(TB_TILE, TB - tb0);
            int slot = (int) (tile_i % CHUNK_DEPTH);
            int pc = phase_of(tile_i);
            cute::wait_barrier(plan.bar_ready[slot], pc);

#pragma unroll
            for (int r = 0; r < TB_TILE; r++)
            {
                if (r >= rows)
                    continue;
                int tb = tb0 + r;
                bf16_t* row_ptr = tile_ptr(slot, r);
                bf16_t* out_ptr = output + (long long) tb * H;
                float sq_local = 0.f;
                float dot_local = 0.f;

#pragma unroll
                for (int si = 0; si < SLICES_PER_GROUP; si++)
                {
                    int dt = si * CONSUMER_GROUPS + group;
                    if (dt >= NHT)
                        continue;
                    int h_base = dt * K_TILE + k_local;
                    float qv[VEC];
                    tmem_ld_32dp32bNx<VEC>(my_tmem + si * VEC, qv);
                    int4 vp = *reinterpret_cast<int4 const*>(row_ptr + h_base);
                    *reinterpret_cast<int4*>(out_ptr + h_base) = vp;

                    __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
                    float2 f0 = __bfloat1622float2(v2[0]);
                    float2 f1 = __bfloat1622float2(v2[1]);
                    float2 f2 = __bfloat1622float2(v2[2]);
                    float2 f3 = __bfloat1622float2(v2[3]);
                    sq_local += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y + f2.x * f2.x + f2.y * f2.y
                        + f3.x * f3.x + f3.y * f3.y;
                    dot_local += f0.x * qv[0] + f0.y * qv[1] + f1.x * qv[2] + f1.y * qv[3] + f2.x * qv[4] + f2.y * qv[5]
                        + f3.x * qv[6] + f3.y * qv[7];
                }

#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                {
                    sq_local += __shfl_xor_sync(0xffffffff, sq_local, offset);
                    dot_local += __shfl_xor_sync(0xffffffff, dot_local, offset);
                }
                if (lane == 0)
                {
                    plan.ws_stats[comp_wid][0] = make_float2(sq_local, dot_local);
                }
                cutlass::arch::NamedBarrier::sync(CONSUMER_THREADS, 0);

                if (comp_wid == 0 && lane == 0)
                {
                    float2 totals = {};
#pragma unroll
                    for (int w = 0; w < CONSUMER_WARPS; w++)
                    {
                        totals = float2_add(totals, plan.ws_stats[w][0]);
                    }
                    float rs = rsqrtf(totals.x / H + rms_eps);
                    rsigma_out[tb] = rs;
                    if (logits_out)
                        logits_out[tb] = totals.y * rs;
                    if (probs_out)
                        probs_out[tb] = 1.f;
                }
                cutlass::arch::NamedBarrier::sync(CONSUMER_THREADS, 1);
            }
            cute::arrive_barrier(plan.bar_consumed[slot]);
        }
    }

    __syncthreads();
    if (wid == 1)
    {
        cute::TMEM::Allocator1Sm alloc;
        alloc.free(plan.tmem_base, TMEM_Q_COLS_TOTAL);
    }
#else
    if (cute::thread0())
        printf("attn_res_fwd_n1_ttile_kernel requires sm_100a\n");
#endif
}

template <int H, int NC = N_CHUNK_DEFAULT, bool RELEASE_TMEM = false, bool FULL_N12 = false>
static void launch_fwd(bf16_t const* block_residual, bf16_t const* layer_residual, bf16_t const* res_weight,
    bf16_t const* rms_weight, bf16_t* output, float* rsigma, float* probs, float* logits, int N, int T, int B,
    float rms_eps, int num_sm, cudaStream_t stream)
{
    constexpr size_t smem_size
        = ((size_t) CHUNK_DEPTH * NC * H * sizeof(bf16_t) + sizeof(FwdSmemPlan<NC>) + 15) & ~size_t(15);
    auto kernel = &attn_res_fwd_online_v2_kernel<H, NC, RELEASE_TMEM, FULL_N12>;
    if (smem_size > 48 * 1024)
    {
        // cudaFuncSetAttribute applies to the current device only; set it
        // once per device (per kernel instantiation).
        static std::once_flag attrs_set[64];
        int dev = 0;
        TLLM_CUDA_CHECK(cudaGetDevice(&dev));
        auto const set_attr = [&]
        { TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)); };
        if (dev >= 0 && dev < 64)
        {
            std::call_once(attrs_set[dev], set_attr);
        }
        else
        {
            set_attr();
        }
    }
    int grid = RELEASE_TMEM ? num_sm * 2 : num_sm;
    kernel<<<grid, BLK, smem_size, stream>>>(
        block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs, logits, N, T, B, rms_eps);
}

// Small-N counterpart to the Triton one-program topology.  One CTA owns the
// complete token, with exactly 28 hidden elements per thread at H=7168.  For
// N=2/4, packed BF16 V remains in registers across the statistics/softmax
// boundary; N=1 can write V directly because its softmax is identically one.
template <int N>
__global__ void __launch_bounds__(256, 1)
    attn_res_fwd_s1_single_cta_kernel(bf16_t const* __restrict__ block_res, bf16_t const* __restrict__ layer_res,
        bf16_t const* __restrict__ res_w, bf16_t const* __restrict__ rms_w, bf16_t* __restrict__ output,
        float* __restrict__ rsigma_out, float* __restrict__ probs_out, float* __restrict__ logits_out, float rms_eps)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    constexpr int H = 7168;
    constexpr int THREADS = 256;
    constexpr int WARPS = THREADS / 32;
    constexpr int ITEMS = H / THREADS;
    constexpr float LOG2_E = 1.4426950408889634f;
    static_assert(H % THREADS == 0);
    static_assert(N == 1 || N == 2 || N == 4);

    __shared__ float2 warp_stats[WARPS * N];
    __shared__ float weights[N];
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;

    float2 stats[N] = {};
    uint32_t v_cache_bf16[ITEMS][(N + 1) / 2];
#pragma unroll
    for (int item = 0; item < ITEMS; item++)
    {
        int h = tid + item * THREADS;
        float q = __bfloat162float(res_w[h]) * __bfloat162float(rms_w[h]);
        bf16_t item_v[N];
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            bf16_t const* row = n < N - 1 ? block_res + (size_t) n * H : layer_res;
            bf16_t packed_v = row[h];
            float v = __bfloat162float(packed_v);
            if constexpr (N == 1)
            {
                output[h] = packed_v;
            }
            else
            {
                item_v[n] = packed_v;
            }
            stats[n] = float2_fma(make_float2(v, v), make_float2(v, q), stats[n]);
        }
        if constexpr (N > 1)
        {
#pragma unroll
            for (int pair = 0; pair < N / 2; pair++)
            {
                union
                {
                    __nv_bfloat162 bf16x2;
                    uint32_t bits;
                } packed;

                packed.bf16x2 = __halves2bfloat162(item_v[2 * pair], item_v[2 * pair + 1]);
                v_cache_bf16[item][pair] = packed.bits;
            }
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            uint64_t packed = reinterpret_cast<uint64_t&>(stats[n]);
            packed = __shfl_down_sync(0xffffffff, packed, offset);
            float2 other = reinterpret_cast<float2&>(packed);
            stats[n] = float2_add(stats[n], other);
        }
    }
    if (lane == 0)
    {
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            warp_stats[warp * N + n] = stats[n];
        }
    }
    __syncthreads();

    if (tid < N)
    {
        float2 total = {};
#pragma unroll
        for (int w = 0; w < WARPS; w++)
        {
            total = float2_add(total, warp_stats[w * N + tid]);
        }
        warp_stats[tid] = total;
    }
    __syncthreads();

    if (tid == 0)
    {
        float local_rsigma[N];
        float local_logits[N];
        float max_logit = -FLT_MAX;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            float2 total = warp_stats[n];
            local_rsigma[n] = rsqrtf(total.x / H + rms_eps);
            local_logits[n] = total.y * local_rsigma[n];
            max_logit = fmaxf(max_logit, local_logits[n]);
        }
        float denominator = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            weights[n] = exp2f((local_logits[n] - max_logit) * LOG2_E);
            denominator += weights[n];
        }
        float inv_denominator = 1.0f / denominator;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            weights[n] *= inv_denominator;
            rsigma_out[n] = local_rsigma[n];
            logits_out[n] = local_logits[n];
            probs_out[n] = weights[n];
        }
    }
    __syncthreads();

    if constexpr (N > 1)
    {
#pragma unroll
        for (int item = 0; item < ITEMS; item++)
        {
            float value = 0.0f;
#pragma unroll
            for (int pair = 0; pair < N / 2; pair++)
            {
                union
                {
                    __nv_bfloat162 bf16x2;
                    uint32_t bits;
                } packed;

                packed.bits = v_cache_bf16[item][pair];
                float2 v = __bfloat1622float2(packed.bf16x2);
                value = fmaf(weights[2 * pair], v.x, value);
                value = fmaf(weights[2 * pair + 1], v.y, value);
            }
            int h = tid + item * THREADS;
            output[h] = __float2bfloat16_rn(value);
        }
    }
#else
    if (cute::thread0())
    {
        printf("attn_res_fwd_s1_single_cta_kernel requires sm_100a\n");
    }
#endif
}

template <int N>
static void launch_s1_single_cta(bf16_t const* block_residual, bf16_t const* layer_residual, bf16_t const* res_weight,
    bf16_t const* rms_weight, bf16_t* output, float* rsigma, float* probs, float* logits, float rms_eps,
    cudaStream_t stream)
{
    attn_res_fwd_s1_single_cta_kernel<N><<<1, 256, 0, stream>>>(
        block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs, logits, rms_eps);
}

// Single-token split-K specialization.  The complete grid is one CTA cluster:
// rank g owns a disjoint H/GROUPS slice, keeps that slice of FP32 V in its
// rank-local shared memory, and exchanges only (square, dot) partials via DSM.
template <int N, int GROUPS = 8>
__global__ void __launch_bounds__(256, 1)
    attn_res_fwd_s1_splitk_kernel(bf16_t const* __restrict__ block_res, bf16_t const* __restrict__ layer_res,
        bf16_t const* __restrict__ res_w, bf16_t const* __restrict__ rms_w, bf16_t* __restrict__ output,
        float* __restrict__ rsigma_out, float* __restrict__ probs_out, float* __restrict__ logits_out, float rms_eps)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    namespace cg = cooperative_groups;
    constexpr int H = 7168;
    constexpr int K_PER_CTA = H / GROUPS;
    constexpr int THREADS = 256;
    constexpr int WARPS = THREADS / 32;
    constexpr float LOG2_E = 1.4426950408889634f;
    static_assert(H % GROUPS == 0);

    extern __shared__ char smem_raw[];
    float* v_cache = reinterpret_cast<float*>(smem_raw);
    float2* warp_stats = reinterpret_cast<float2*>(smem_raw + (size_t) N * K_PER_CTA * sizeof(float));
    float* weights = reinterpret_cast<float*>(warp_stats + WARPS * N);

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    cg::cluster_group cluster = cg::this_cluster();
    int const group = cluster.block_rank();
    int const h_begin = group * K_PER_CTA;

    float sq[N] = {};
    float dot[N] = {};
#pragma unroll
    for (int ki = tid; ki < K_PER_CTA; ki += THREADS)
    {
        int h = h_begin + ki;
        float q = __bfloat162float(res_w[h]) * __bfloat162float(rms_w[h]);
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            bf16_t const* row = n < N - 1 ? block_res + (size_t) n * H : layer_res;
            float v = __bfloat162float(row[h]);
            v_cache[(size_t) n * K_PER_CTA + ki] = v;
            sq[n] = fmaf(v, v, sq[n]);
            dot[n] = fmaf(v, q, dot[n]);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            sq[n] += __shfl_down_sync(0xffffffff, sq[n], offset);
            dot[n] += __shfl_down_sync(0xffffffff, dot[n], offset);
        }
    }
    if (lane == 0)
    {
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            warp_stats[warp * N + n] = make_float2(sq[n], dot[n]);
        }
    }
    __syncthreads();

    if (tid < N)
    {
        float2 total = {};
#pragma unroll
        for (int w = 0; w < WARPS; w++)
        {
            total = float2_add(total, warp_stats[w * N + tid]);
        }
        warp_stats[tid] = total;
    }

    // Publish every rank's reduced statistics to distributed shared memory.
    cluster.sync();

    // One thread per candidate reduces across CTA ranks.  Parallelizing this
    // avoids making a single leader issue all GROUPS*N remote DSM reads.
    if (tid < N)
    {
        float2 total = {};
#pragma unroll
        for (int g = 0; g < GROUPS; g++)
        {
            float2 const* remote_stats = cluster.map_shared_rank(warp_stats, g);
            total = float2_add(total, remote_stats[tid]);
        }
        warp_stats[tid] = total;
    }
    __syncthreads();

    if (tid == 0)
    {
        float local_rsigma[N];
        float local_logits[N];
        float max_logit = -FLT_MAX;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            float2 total = warp_stats[n];
            local_rsigma[n] = rsqrtf(total.x / H + rms_eps);
            local_logits[n] = total.y * local_rsigma[n];
            max_logit = fmaxf(max_logit, local_logits[n]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            weights[n] = exp2f((local_logits[n] - max_logit) * LOG2_E);
            sum += weights[n];
        }
        float inv_sum = 1.0f / sum;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            weights[n] *= inv_sum;
            if (group == 0)
            {
                rsigma_out[n] = local_rsigma[n];
                logits_out[n] = local_logits[n];
                probs_out[n] = weights[n];
            }
        }
    }

    cluster.sync();

#pragma unroll
    for (int ki = tid; ki < K_PER_CTA; ki += THREADS)
    {
        float value = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            value = fmaf(weights[n], v_cache[(size_t) n * K_PER_CTA + ki], value);
        }
        output[h_begin + ki] = __float2bfloat16_rn(value);
    }
#else
    if (cute::thread0())
    {
        printf("attn_res_fwd_s1_splitk_kernel requires sm_100a\n");
    }
#endif
}

template <int N, int GROUPS = 8>
static void launch_s1_splitk(bf16_t const* block_residual, bf16_t const* layer_residual, bf16_t const* res_weight,
    bf16_t const* rms_weight, bf16_t* output, float* rsigma, float* probs, float* logits, float rms_eps,
    cudaStream_t stream)
{
    constexpr int K_PER_CTA = 7168 / GROUPS;
    constexpr int WARPS = 8;
    constexpr size_t smem_size
        = (size_t) N * K_PER_CTA * sizeof(float) + (size_t) WARPS * N * sizeof(float2) + (size_t) N * sizeof(float);
    auto kernel = &attn_res_fwd_s1_splitk_kernel<N, GROUPS>;
    {
        // cudaFuncSetAttribute applies to the current device only; set it
        // once per device (per kernel instantiation).
        static std::once_flag attrs_set[64];
        int dev = 0;
        TLLM_CUDA_CHECK(cudaGetDevice(&dev));
        auto const set_attr = [&]
        { TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)); };
        if (dev >= 0 && dev < 64)
        {
            std::call_once(attrs_set[dev], set_attr);
        }
        else
        {
            set_attr();
        }
    }
    void* args[] = {const_cast<bf16_t**>(&block_residual), const_cast<bf16_t**>(&layer_residual),
        const_cast<bf16_t**>(&res_weight), const_cast<bf16_t**>(&rms_weight), &output, &rsigma, &probs, &logits,
        &rms_eps};
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(GROUPS);
    config.blockDim = dim3(256);
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = GROUPS;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    cudaLaunchKernelExC(&config, reinterpret_cast<void const*>(kernel), args);
}

template <int H, int TB_TILE>
static void launch_n1_ttile(bf16_t const* layer_residual, bf16_t const* res_weight, bf16_t const* rms_weight,
    bf16_t* output, float* rsigma, float* probs, float* logits, int T, int B, float rms_eps, int num_sm,
    cudaStream_t stream)
{
    constexpr size_t smem_size
        = ((size_t) CHUNK_DEPTH * TB_TILE * H * sizeof(bf16_t) + sizeof(FwdSmemPlan<1>) + 15) & ~size_t(15);
    auto kernel = &attn_res_fwd_n1_ttile_kernel<H, TB_TILE, true>;
    if (smem_size > 48 * 1024)
    {
        // cudaFuncSetAttribute applies to the current device only; set it
        // once per device (per kernel instantiation).
        static std::once_flag attrs_set[64];
        int dev = 0;
        TLLM_CUDA_CHECK(cudaGetDevice(&dev));
        auto const set_attr = [&]
        { TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)); };
        if (dev >= 0 && dev < 64)
        {
            std::call_once(attrs_set[dev], set_attr);
        }
        else
        {
            set_attr();
        }
    }
    kernel<<<num_sm * 2, BLK, smem_size, stream>>>(
        layer_residual, res_weight, rms_weight, output, rsigma, probs, logits, T, B, rms_eps);
}

} // namespace fwd_prod_v2
} // namespace sm100

int attn_res_fwd_grid_size(int dev)
{
    static int cached_num_sm[64] = {};
    if (dev >= 0 && dev < 64 && cached_num_sm[dev] > 0)
    {
        return cached_num_sm[dev];
    }
    int n = 0;
    cudaError_t err = cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess || n <= 0)
        return 0;
    if (dev >= 0 && dev < 64)
    {
        cached_num_sm[dev] = n;
    }
    return n;
}

} // namespace

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimiK3AttnRes
{

void invokeAttnResFwd(AttnResFwdParams const& params, cudaStream_t stream)
{
    using namespace sm100::fwd_prod_v2;

    bf16_t const* block_residual = params.blockResidual;
    bf16_t const* layer_residual = params.layerResidual;
    bf16_t const* res_weight = params.resWeight;
    bf16_t const* rms_weight = params.rmsWeight;
    bf16_t* output = params.output;
    float* rsigma = params.rsigma;
    float* probs = params.probs;
    float* logits = params.logits;
    int const N = params.numCandidates;
    int const T = params.seqLen;
    int const B = params.batchSize;
    int const H = params.hiddenSize;
    float const rms_eps = params.rmsEps;

    int dev = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&dev));
    int num_sm = attn_res_fwd_grid_size(dev);
    TLLM_CHECK_WITH_INFO(num_sm > 0, "attn_res_fwd: failed to query the SM count of device %d", dev);
    TLLM_CHECK_WITH_INFO(N <= N_MAX, "attn_res_fwd: unsupported N=%d (max %d)", N, N_MAX);

    if (H == 8192)
    {
        if (N == 1)
        {
            launch_n1_ttile<8192, 2>(
                layer_residual, res_weight, rms_weight, output, rsigma, probs, logits, T, B, rms_eps, num_sm, stream);
        }
        else if (N <= 2)
        {
            launch_fwd<8192, 2, true>(block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs,
                logits, N, T, B, rms_eps, num_sm, stream);
        }
        else
        {
            launch_fwd<8192, 4, false>(block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs,
                logits, N, T, B, rms_eps, num_sm, stream);
        }
    }
    else if (H == 7168)
    {
        if (T == 1 && N == 1)
        {
            launch_s1_single_cta<1>(
                block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs, logits, rms_eps, stream);
        }
        else if (T == 1 && N == 2)
        {
            launch_s1_single_cta<2>(
                block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs, logits, rms_eps, stream);
        }
        else if (T == 1 && N == 4)
        {
            launch_s1_single_cta<4>(
                block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs, logits, rms_eps, stream);
        }
        else if (T == 1 && N == 8)
        {
            launch_s1_splitk<8, 8>(
                block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs, logits, rms_eps, stream);
        }
        else if (T == 1 && N == 12)
        {
            launch_s1_splitk<12, 8>(
                block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs, logits, rms_eps, stream);
        }
        else if (N == 12 && T == 1024)
        {
            launch_fwd<7168, 4, false, true>(block_residual, layer_residual, res_weight, rms_weight, output, rsigma,
                probs, logits, N, T, B, rms_eps, std::max(1, num_sm - 1), stream);
        }
        else
        {
            launch_fwd<7168, 4, false>(block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs,
                logits, N, T, B, rms_eps, num_sm, stream);
        }
    }
    else if (H == 6144)
    {
        launch_fwd<6144, 4, false>(block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs,
            logits, N, T, B, rms_eps, num_sm, stream);
    }
    else if (H == 5120)
    {
        launch_fwd<5120, 4, false>(block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs,
            logits, N, T, B, rms_eps, num_sm, stream);
    }
    else if (H == 4096)
    {
        if (N == 1)
        {
            launch_n1_ttile<4096, 4>(
                layer_residual, res_weight, rms_weight, output, rsigma, probs, logits, T, B, rms_eps, num_sm, stream);
        }
        else if (N <= 4)
        {
            launch_fwd<4096, 2, true>(block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs,
                logits, N, T, B, rms_eps, num_sm, stream);
        }
        else
        {
            launch_fwd<4096, 3, true>(block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs,
                logits, N, T, B, rms_eps, num_sm, stream);
        }
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "attn_res_fwd: unsupported hidden size H=%d", H);
    }
}

} // namespace kernels::kimiK3AttnRes

TRTLLM_NAMESPACE_END
