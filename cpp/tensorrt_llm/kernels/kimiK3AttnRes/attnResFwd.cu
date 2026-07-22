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
//   - Q=res_weight*rms_weight is staged in TMEM.
//   - V rows are cached in registers between Pass A and Pass B.
//
// Contract: B=1, N<=12, H in [4096,8192] and divisible by 1024; checked at
// the Torch-op bridge. The kernel uses separate layer/block residual inputs
// and does not require a concatenated V tensor.
//
// Source-integrated from the NVIDIA+Moonshot jointly developed
// Attention_residual kernel (fwd-only b2091dc implementation).

#include "tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.h"

#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

namespace
{

using bf16_t = __nv_bfloat16;


static constexpr int ATTN_RES_BLOCK = 256;
static constexpr int ATTN_RES_WARPS = ATTN_RES_BLOCK / 32;

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_sum(float val, float* ws) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) ws[wid] = val;
    __syncthreads();
    val = (threadIdx.x < ATTN_RES_WARPS) ? ws[threadIdx.x] : 0.f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ __forceinline__
const bf16_t* v_addr(const bf16_t* block_res, const bf16_t* layer_res,
                     int n, int N, int t, int b, int T, int B, int H) {
    if (n < N - 1)
        return block_res + (((long long)n * T + t) * B + b) * H;
    return layer_res + ((long long)t * B + b) * H;
}

namespace sm100 {


CUTE_DEVICE
void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

CUTE_DEVICE
void umma_arrive_noelect(uint64_t& bar_ptr) {
    uint64_t bar_addr = cute::cast_smem_ptr_to_uint(&bar_ptr);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :
        : "l"(bar_addr));
}

CUTE_DEVICE
float2 float2_sub(const float2& a, const float2& b) {
    float2 c;
    asm volatile(
        "sub.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

CUTE_DEVICE
float2 float2_mul(const float2& a, const float2& b) {
    float2 c;
    asm volatile(
        "mul.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

CUTE_DEVICE
float2 float2_fma(const float2& a, const float2& b, const float2& c) {
    float2 d;
    asm volatile(
        "fma.rn.f32x2 %0, %1, %2, %3;\n"
        : "=l"(reinterpret_cast<uint64_t&>(d))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)),
          "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

template <int N, typename T>
CUTE_DEVICE void tmem_ld_32dp32bNx(uint32_t const& src_addr, T* dst_ptr_) {
    static_assert(N == 8, "attn_res production TMEM helpers only instantiate x8");
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst_ptr_);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        "{%0, %1, %2, %3, %4, %5, %6, %7},"
        "[%8];\n"
        : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
          "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
          "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
        : "r"(src_addr));
}

template <int N, typename T>
CUTE_DEVICE void tmem_st_32dp32bNx(uint32_t const& dst_addr, T* src_ptr_) {
    static_assert(N == 8, "attn_res production TMEM helpers only instantiate x8");
    uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src_ptr_);
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        "[%8], {%0, %1, %2, %3, %4, %5, %6, %7};\n"
        :
        : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
          "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
          "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(dst_addr));
}


namespace fwd_prod_v2 {

using namespace cute;

constexpr int K_TILE = 1024;
constexpr int N_MAX = 12;
constexpr int N_CHUNK_DEFAULT = 4;
constexpr int CHUNK_DEPTH = 2;
constexpr int BLK = 288;                         // 1 producer warp + 8 consumer warps
constexpr int CONSUMER_THREADS = BLK - 32;       // 256
constexpr int CONSUMER_WARPS = CONSUMER_THREADS / 32;
constexpr int CONSUMER_GROUPS = 2;               // two 128-thread consumer groups
constexpr int CONSUMER_THREADS_PER_GROUP = CONSUMER_THREADS / CONSUMER_GROUPS;
constexpr int TMEM_Q_COLS_PER_GROUP = 32;
constexpr int TMEM_Q_COLS_TOTAL = 2 * TMEM_Q_COLS_PER_GROUP;

template <int NC>
struct FwdSmemPlan {
    alignas(16) uint64_t bar_ready[CHUNK_DEPTH];
    alignas(16) uint64_t bar_consumed[CHUNK_DEPTH];
    alignas(16) float ws_sq[CONSUMER_WARPS][NC];
    alignas(16) float ws_dot[CONSUMER_WARPS][NC];
    alignas(16) float rsigma_ch[NC];
    alignas(16) float logit_ch[NC];
    uint32_t tmem_base;
};

__device__ __forceinline__
void cp_async_bulk(void* smem_dst, const void* gmem_src, int bytes, uint64_t& mbar) {
    uint32_t s = cute::cast_smem_ptr_to_uint(smem_dst);
    uint32_t m = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(s), "l"(gmem_src), "r"(bytes), "r"(m) : "memory");
}

template <int H, int NC = N_CHUNK_DEFAULT, bool RELEASE_TMEM = false>
__global__ void __launch_bounds__(BLK, 1)
attn_res_fwd_online_v2_kernel(
    const bf16_t* __restrict__ block_res,
    const bf16_t* __restrict__ layer_res,
    const bf16_t* __restrict__ res_w,
    const bf16_t* __restrict__ rms_w,
    bf16_t* __restrict__ output,
    float* __restrict__ rsigma_out,
    float* __restrict__ probs_out,
    float* __restrict__ logits_out,
    int N, int T, int B, float rms_eps)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    constexpr float LOG2_E = 1.4426950408889634f;
    constexpr int N_CHUNK = NC;
    constexpr int NUM_BUFS = CHUNK_DEPTH * NC;
    constexpr int NHT = H / K_TILE;
    constexpr int SLICES_PER_GROUP = (NHT + CONSUMER_GROUPS - 1) / CONSUMER_GROUPS;
    constexpr int VEC = 8;
    constexpr int ACC_PER_THREAD = SLICES_PER_GROUP * VEC;
    static_assert(H >= 4096 && H <= 8192);
    static_assert(H % K_TILE == 0);

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int TB = T * B;
    const int num_ctas = gridDim.x;
    const int num_chunks = (N + N_CHUNK - 1) / N_CHUNK;

    const int comp_wid = wid - 1;
    const int comp_tid = tid - 32;
    const int group = (comp_wid >= 4) ? 1 : 0;
    const int ct_in_group = (comp_tid >= 0) ? (comp_tid & (CONSUMER_THREADS_PER_GROUP - 1)) : -1;
    const int k_local = ct_in_group * VEC;

    extern __shared__ char smem_raw[];
    bf16_t* v_bufs = reinterpret_cast<bf16_t*>(smem_raw);   // [NUM_BUFS][H]
    constexpr size_t V_BYTES = (size_t)NUM_BUFS * H * sizeof(bf16_t);
    FwdSmemPlan<NC>& plan = *reinterpret_cast<FwdSmemPlan<NC>*>(smem_raw + V_BYTES);

    auto slot_of = [](long long gci, int n) {
        return (int)(gci % CHUNK_DEPTH) * N_CHUNK + n;
    };
    auto phase_of = [](long long gci) {
        return (int)((gci / CHUNK_DEPTH) & 1);
    };
    auto buf_ptr = [&](int slot) -> bf16_t* {
        return v_bufs + slot * H;
    };

    if (wid == 0 && elect_one_sync()) {
        #pragma unroll
        for (int i = 0; i < CHUNK_DEPTH; i++) {
            cute::initialize_barrier(plan.bar_ready[i], 1);
            cute::initialize_barrier(plan.bar_consumed[i], CONSUMER_THREADS);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (wid == 1) {
        cute::TMEM::Allocator1Sm alloc;
        alloc.allocate(TMEM_Q_COLS_TOTAL, &plan.tmem_base);
        if constexpr (RELEASE_TMEM) {
            alloc.release_allocation_lock();
        }
    }
    __syncthreads();

    const uint32_t my_tmem = (comp_tid >= 0)
        ? (plan.tmem_base + ((comp_wid >= 4) ? TMEM_Q_COLS_PER_GROUP : 0))
        : 0;

    if (comp_tid >= 0) {
        float q32[ACC_PER_THREAD];
        #pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++) {
            int dt = si * CONSUMER_GROUPS + group;
            if (dt >= NHT) continue;
            int h_base = dt * K_TILE + k_local;
            #pragma unroll
            for (int j = 0; j < VEC; j++) {
                int h = h_base + j;
                q32[si * VEC + j] =
                    __bfloat162float(rms_w[h]) * __bfloat162float(res_w[h]);
            }
        }
        #pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++) {
            tmem_st_32dp32bNx<VEC>(my_tmem + si * VEC, &q32[si * VEC]);
        }
        cutlass::arch::fence_view_async_tmem_store();
    }
    __syncthreads();

    if (wid == 0) {
        if (elect_one_sync()) {
            long long gci = 0;
            for (int tb = blockIdx.x; tb < TB; tb += num_ctas) {
                const int t = tb / B;
                const int b = tb % B;
                for (int ci = 0; ci < num_chunks; ci++, gci++) {
                    int ns = ci * N_CHUNK;
                    int an = min(N_CHUNK, N - ns);
                    int chunk_slot = (int)(gci % CHUNK_DEPTH);
                    int pc = phase_of(gci);
                    cute::wait_barrier(plan.bar_consumed[chunk_slot], pc ^ 1);
                    cute::set_barrier_transaction_bytes(
                        plan.bar_ready[chunk_slot], an * H * (int)sizeof(bf16_t));
                    #pragma unroll
                    for (int n = 0; n < N_CHUNK; n++) {
                        if (n >= an) continue;
                        int slot = slot_of(gci, n);
                        const bf16_t* src = v_addr(
                            block_res, layer_res, ns + n, N, t, b, T, B, H);
                        cp_async_bulk(buf_ptr(slot), src, H * sizeof(bf16_t),
                                      plan.bar_ready[chunk_slot]);
                    }
                }
            }
        }
    } else {
        float acc32[ACC_PER_THREAD] = {};

        long long gci = 0;
        for (int tb = blockIdx.x; tb < TB; tb += num_ctas) {
            float m_running = -FLT_MAX;
            float s_running = 0.f;
            #pragma unroll
            for (int i = 0; i < ACC_PER_THREAD; i++) {
                acc32[i] = 0.f;
            }

            for (int ci = 0; ci < num_chunks; ci++, gci++) {
                int ns = ci * N_CHUNK;
                int an = min(N_CHUNK, N - ns);
                int chunk_slot = (int)(gci % CHUNK_DEPTH);
                int pr = phase_of(gci);
                cute::wait_barrier(plan.bar_ready[chunk_slot], pr);

                float sq_local[N_CHUNK] = {};
                float dot_local[N_CHUNK] = {};
                int4 v_cache[SLICES_PER_GROUP][N_CHUNK];

                auto pass_A_body = [&](auto AN_TOK) {
                    constexpr int AN = decltype(AN_TOK)::value;
                    #pragma unroll
                    for (int si = 0; si < SLICES_PER_GROUP; si++) {
                        int dt = si * CONSUMER_GROUPS + group;
                        if (dt >= NHT) continue;
                        float qv[VEC];
                        tmem_ld_32dp32bNx<VEC>(my_tmem + si * VEC, qv);

                        #pragma unroll
                        for (int n = 0; n < AN; n++) {
                            int slot = slot_of(gci, n);
                            int4 vp = *reinterpret_cast<const int4*>(
                                buf_ptr(slot) + dt * K_TILE + k_local);
                            v_cache[si][n] = vp;
                            __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
                            float2 f0 = __bfloat1622float2(v2[0]);
                            float2 f1 = __bfloat1622float2(v2[1]);
                            float2 f2 = __bfloat1622float2(v2[2]);
                            float2 f3 = __bfloat1622float2(v2[3]);
                            sq_local[n] +=
                                f0.x * f0.x + f0.y * f0.y +
                                f1.x * f1.x + f1.y * f1.y +
                                f2.x * f2.x + f2.y * f2.y +
                                f3.x * f3.x + f3.y * f3.y;
                            dot_local[n] +=
                                f0.x * qv[0] + f0.y * qv[1] +
                                f1.x * qv[2] + f1.y * qv[3] +
                                f2.x * qv[4] + f2.y * qv[5] +
                                f3.x * qv[6] + f3.y * qv[7];
                        }
                    }
                };
                if constexpr (NC == 4) {
                    switch (an) {
                        case 4: pass_A_body(std::integral_constant<int, 4>{}); break;
                        case 3: pass_A_body(std::integral_constant<int, 3>{}); break;
                        case 2: pass_A_body(std::integral_constant<int, 2>{}); break;
                        case 1: pass_A_body(std::integral_constant<int, 1>{}); break;
                        default: __builtin_unreachable();
                    }
                } else if constexpr (NC == 3) {
                    switch (an) {
                        case 3: pass_A_body(std::integral_constant<int, 3>{}); break;
                        case 2: pass_A_body(std::integral_constant<int, 2>{}); break;
                        case 1: pass_A_body(std::integral_constant<int, 1>{}); break;
                        default: __builtin_unreachable();
                    }
                } else {
                    static_assert(NC == 2);
                    switch (an) {
                        case 2: pass_A_body(std::integral_constant<int, 2>{}); break;
                        case 1: pass_A_body(std::integral_constant<int, 1>{}); break;
                        default: __builtin_unreachable();
                    }
                }
                cute::arrive_barrier(plan.bar_consumed[chunk_slot]);

                float sq[N_CHUNK], dot[N_CHUNK];
                #pragma unroll
                for (int n = 0; n < N_CHUNK; n++) {
                    sq[n] = sq_local[n];
                    dot[n] = dot_local[n];
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    #pragma unroll
                    for (int n = 0; n < N_CHUNK; n++) {
                        sq[n] += __shfl_xor_sync(0xffffffff, sq[n], offset);
                        dot[n] += __shfl_xor_sync(0xffffffff, dot[n], offset);
                    }
                }
                if (lane == 0) {
                    #pragma unroll
                    for (int n = 0; n < N_CHUNK; n++) {
                        plan.ws_sq[comp_wid][n] = sq[n];
                        plan.ws_dot[comp_wid][n] = dot[n];
                    }
                }
                cutlass::arch::NamedBarrier::sync(CONSUMER_THREADS, 0);

                float local_rsig = 0.f;
                float local_logit = 0.f;
                if (lane < N_CHUNK) {
                    int n = lane;
                    float sqs = 0.f;
                    float dotss = 0.f;
                    #pragma unroll
                    for (int w = 0; w < CONSUMER_WARPS; w++) {
                        sqs += plan.ws_sq[w][n];
                        dotss += plan.ws_dot[w][n];
                    }
                    local_rsig = rsqrtf(sqs / H + rms_eps);
                    local_logit = dotss * local_rsig;
                }
                float rsig_n[N_CHUNK];
                float logit_n[N_CHUNK];
                #pragma unroll
                for (int n = 0; n < N_CHUNK; n++) {
                    rsig_n[n] = __shfl_sync(0xffffffff, local_rsig, n);
                    logit_n[n] = __shfl_sync(0xffffffff, local_logit, n);
                }

                float m_chunk = -FLT_MAX;
                #pragma unroll
                for (int n = 0; n < N_CHUNK; n++) {
                    if (n < an) m_chunk = fmaxf(m_chunk, logit_n[n]);
                }
                float m_new = fmaxf(m_running, m_chunk);
                float corr = exp2f((m_running - m_new) * LOG2_E);
                float w_n[N_CHUNK] = {};
                float w_sum = 0.f;
                #pragma unroll
                for (int n = 0; n < N_CHUNK; n++) {
                    if (n < an) {
                        w_n[n] = exp2f((logit_n[n] - m_new) * LOG2_E);
                        w_sum += w_n[n];
                    }
                }

                auto pass_B_body = [&](auto AN_TOK) {
                    constexpr int AN = decltype(AN_TOK)::value;
                    #pragma unroll
                    for (int si = 0; si < SLICES_PER_GROUP; si++) {
                        int dt = si * CONSUMER_GROUPS + group;
                        if (dt >= NHT) continue;
                        float a[VEC];
                        #pragma unroll
                        for (int j = 0; j < VEC; j++) {
                            a[j] = acc32[si * VEC + j] * corr;
                        }
                        #pragma unroll
                        for (int n = 0; n < AN; n++) {
                            int4 vp = v_cache[si][n];
                            __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
                            float2 f0 = __bfloat1622float2(v2[0]);
                            float2 f1 = __bfloat1622float2(v2[1]);
                            float2 f2 = __bfloat1622float2(v2[2]);
                            float2 f3 = __bfloat1622float2(v2[3]);
                            float wn = w_n[n];
                            a[0] += wn * f0.x; a[1] += wn * f0.y;
                            a[2] += wn * f1.x; a[3] += wn * f1.y;
                            a[4] += wn * f2.x; a[5] += wn * f2.y;
                            a[6] += wn * f3.x; a[7] += wn * f3.y;
                        }
                        #pragma unroll
                        for (int j = 0; j < VEC; j++) {
                            acc32[si * VEC + j] = a[j];
                        }
                    }
                };
                if constexpr (NC == 4) {
                    switch (an) {
                        case 4: pass_B_body(std::integral_constant<int, 4>{}); break;
                        case 3: pass_B_body(std::integral_constant<int, 3>{}); break;
                        case 2: pass_B_body(std::integral_constant<int, 2>{}); break;
                        case 1: pass_B_body(std::integral_constant<int, 1>{}); break;
                        default: __builtin_unreachable();
                    }
                } else if constexpr (NC == 3) {
                    switch (an) {
                        case 3: pass_B_body(std::integral_constant<int, 3>{}); break;
                        case 2: pass_B_body(std::integral_constant<int, 2>{}); break;
                        case 1: pass_B_body(std::integral_constant<int, 1>{}); break;
                        default: __builtin_unreachable();
                    }
                } else {
                    static_assert(NC == 2);
                    switch (an) {
                        case 2: pass_B_body(std::integral_constant<int, 2>{}); break;
                        case 1: pass_B_body(std::integral_constant<int, 1>{}); break;
                        default: __builtin_unreachable();
                    }
                }

                s_running = s_running * corr + w_sum;
                m_running = m_new;

                if (comp_wid == 0 && lane < an) {
                    int ng = ns + lane;
                    rsigma_out[(long long)ng * TB + tb] = local_rsig;
                    if (logits_out) {
                        logits_out[(long long)ng * TB + tb] = local_logit;
                    }
                }
            }

            float inv_s = 1.f / s_running;
            bf16_t* out_ptr = output + (long long)tb * H;
            #pragma unroll
            for (int si = 0; si < SLICES_PER_GROUP; si++) {
                int dt = si * CONSUMER_GROUPS + group;
                if (dt >= NHT) continue;
                int h_base = dt * K_TILE + k_local;
                bf16_t ov[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; j++) {
                    ov[j] = __float2bfloat16(acc32[si * VEC + j] * inv_s);
                }
                *reinterpret_cast<int4*>(out_ptr + h_base) =
                    *reinterpret_cast<int4*>(ov);
            }

            if (comp_tid == 0 && probs_out) {
                float mx = m_running;
                for (int n = 0; n < N; n++) {
                    float lg = logits_out[(long long)n * TB + tb];
                    probs_out[(long long)n * TB + tb] =
                        exp2f((lg - mx) * LOG2_E) * inv_s;
                }
            }
        }
    }

    __syncthreads();
    if (wid == 1) {
        cute::TMEM::Allocator1Sm alloc;
        alloc.free(plan.tmem_base, TMEM_Q_COLS_TOTAL);
    }
#else
    if (cute::thread0()) printf("attn_res_fwd_online_v2_kernel requires sm_100a\n");
#endif
}

// N=1 specialization: softmax is degenerate, so output is the layer row.
// Tile multiple contiguous TB rows per CTA to reduce cp.async.bulk overhead.
template <int H, int TB_TILE, bool RELEASE_TMEM = true>
__global__ void __launch_bounds__(BLK, 1)
attn_res_fwd_n1_ttile_kernel(
    const bf16_t* __restrict__ layer_res,
    const bf16_t* __restrict__ res_w,
    const bf16_t* __restrict__ rms_w,
    bf16_t* __restrict__ output,
    float* __restrict__ rsigma_out,
    float* __restrict__ probs_out,
    float* __restrict__ logits_out,
    int T, int B, float rms_eps)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    constexpr int NHT = H / K_TILE;
    constexpr int SLICES_PER_GROUP = (NHT + CONSUMER_GROUPS - 1) / CONSUMER_GROUPS;
    constexpr int VEC = 8;
    constexpr int ACC_PER_THREAD = SLICES_PER_GROUP * VEC;
    static_assert(H == 4096 || H == 8192);

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int TB = T * B;
    const int comp_wid = wid - 1;
    const int comp_tid = tid - 32;
    const int group = (comp_wid >= 4) ? 1 : 0;
    const int ct_in_group = (comp_tid >= 0) ? (comp_tid & (CONSUMER_THREADS_PER_GROUP - 1)) : -1;
    const int k_local = ct_in_group * VEC;

    extern __shared__ char smem_raw[];
    bf16_t* v_tiles = reinterpret_cast<bf16_t*>(smem_raw);
    constexpr size_t V_BYTES = (size_t)CHUNK_DEPTH * TB_TILE * H * sizeof(bf16_t);
    FwdSmemPlan<1>& plan = *reinterpret_cast<FwdSmemPlan<1>*>(smem_raw + V_BYTES);

    auto phase_of = [](long long tile_i) {
        return (int)((tile_i / CHUNK_DEPTH) & 1);
    };
    auto tile_ptr = [&](int slot, int row) -> bf16_t* {
        return v_tiles + ((slot * TB_TILE + row) * H);
    };

    if (wid == 0 && elect_one_sync()) {
        #pragma unroll
        for (int i = 0; i < CHUNK_DEPTH; i++) {
            cute::initialize_barrier(plan.bar_ready[i], 1);
            cute::initialize_barrier(plan.bar_consumed[i], CONSUMER_THREADS);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (wid == 1) {
        cute::TMEM::Allocator1Sm alloc;
        alloc.allocate(TMEM_Q_COLS_TOTAL, &plan.tmem_base);
        if constexpr (RELEASE_TMEM) {
            alloc.release_allocation_lock();
        }
    }
    __syncthreads();

    const uint32_t my_tmem = (comp_tid >= 0)
        ? (plan.tmem_base + ((comp_wid >= 4) ? TMEM_Q_COLS_PER_GROUP : 0))
        : 0;

    if (comp_tid >= 0) {
        float q32[ACC_PER_THREAD];
        #pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++) {
            int dt = si * CONSUMER_GROUPS + group;
            if (dt >= NHT) continue;
            int h_base = dt * K_TILE + k_local;
            #pragma unroll
            for (int j = 0; j < VEC; j++) {
                int h = h_base + j;
                q32[si * VEC + j] =
                    __bfloat162float(rms_w[h]) * __bfloat162float(res_w[h]);
            }
        }
        #pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++) {
            int dt = si * CONSUMER_GROUPS + group;
            if (dt >= NHT) continue;
            tmem_st_32dp32bNx<VEC>(my_tmem + si * VEC, &q32[si * VEC]);
        }
        cutlass::arch::fence_view_async_tmem_store();
    }
    __syncthreads();

    if (wid == 0) {
        if (elect_one_sync()) {
            long long tile_i = 0;
            for (int tb0 = blockIdx.x * TB_TILE; tb0 < TB;
                 tb0 += gridDim.x * TB_TILE, tile_i++) {
                int rows = min(TB_TILE, TB - tb0);
                int slot = (int)(tile_i % CHUNK_DEPTH);
                int pc = phase_of(tile_i);
                cute::wait_barrier(plan.bar_consumed[slot], pc ^ 1);
                cute::set_barrier_transaction_bytes(
                    plan.bar_ready[slot], rows * H * (int)sizeof(bf16_t));
                cp_async_bulk(
                    tile_ptr(slot, 0),
                    layer_res + (long long)tb0 * H,
                    rows * H * sizeof(bf16_t),
                    plan.bar_ready[slot]);
            }
        }
    } else {
        long long tile_i = 0;
        for (int tb0 = blockIdx.x * TB_TILE; tb0 < TB;
             tb0 += gridDim.x * TB_TILE, tile_i++) {
            int rows = min(TB_TILE, TB - tb0);
            int slot = (int)(tile_i % CHUNK_DEPTH);
            int pc = phase_of(tile_i);
            cute::wait_barrier(plan.bar_ready[slot], pc);

            #pragma unroll
            for (int r = 0; r < TB_TILE; r++) {
                if (r >= rows) continue;
                int tb = tb0 + r;
                bf16_t* row_ptr = tile_ptr(slot, r);
                bf16_t* out_ptr = output + (long long)tb * H;
                float sq_local = 0.f;
                float dot_local = 0.f;

                #pragma unroll
                for (int si = 0; si < SLICES_PER_GROUP; si++) {
                    int dt = si * CONSUMER_GROUPS + group;
                    if (dt >= NHT) continue;
                    int h_base = dt * K_TILE + k_local;
                    float qv[VEC];
                    tmem_ld_32dp32bNx<VEC>(my_tmem + si * VEC, qv);
                    int4 vp = *reinterpret_cast<const int4*>(row_ptr + h_base);
                    *reinterpret_cast<int4*>(out_ptr + h_base) = vp;

                    __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
                    float2 f0 = __bfloat1622float2(v2[0]);
                    float2 f1 = __bfloat1622float2(v2[1]);
                    float2 f2 = __bfloat1622float2(v2[2]);
                    float2 f3 = __bfloat1622float2(v2[3]);
                    sq_local +=
                        f0.x * f0.x + f0.y * f0.y +
                        f1.x * f1.x + f1.y * f1.y +
                        f2.x * f2.x + f2.y * f2.y +
                        f3.x * f3.x + f3.y * f3.y;
                    dot_local +=
                        f0.x * qv[0] + f0.y * qv[1] +
                        f1.x * qv[2] + f1.y * qv[3] +
                        f2.x * qv[4] + f2.y * qv[5] +
                        f3.x * qv[6] + f3.y * qv[7];
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    sq_local += __shfl_xor_sync(0xffffffff, sq_local, offset);
                    dot_local += __shfl_xor_sync(0xffffffff, dot_local, offset);
                }
                if (lane == 0) {
                    plan.ws_sq[comp_wid][0] = sq_local;
                    plan.ws_dot[comp_wid][0] = dot_local;
                }
                cutlass::arch::NamedBarrier::sync(CONSUMER_THREADS, 0);

                if (comp_wid == 0 && lane == 0) {
                    float sq = 0.f;
                    float dot = 0.f;
                    #pragma unroll
                    for (int w = 0; w < CONSUMER_WARPS; w++) {
                        sq += plan.ws_sq[w][0];
                        dot += plan.ws_dot[w][0];
                    }
                    float rs = rsqrtf(sq / H + rms_eps);
                    rsigma_out[tb] = rs;
                    if (logits_out) logits_out[tb] = dot * rs;
                    if (probs_out) probs_out[tb] = 1.f;
                }
                cutlass::arch::NamedBarrier::sync(CONSUMER_THREADS, 1);
            }
            cute::arrive_barrier(plan.bar_consumed[slot]);
        }
    }

    __syncthreads();
    if (wid == 1) {
        cute::TMEM::Allocator1Sm alloc;
        alloc.free(plan.tmem_base, TMEM_Q_COLS_TOTAL);
    }
#else
    if (cute::thread0()) printf("attn_res_fwd_n1_ttile_kernel requires sm_100a\n");
#endif
}

template <int H, int NC = N_CHUNK_DEFAULT, bool RELEASE_TMEM = false>
static void launch_fwd(
    const bf16_t* block_residual,
    const bf16_t* layer_residual,
    const bf16_t* res_weight,
    const bf16_t* rms_weight,
    bf16_t* output,
    float* rsigma,
    float* probs,
    float* logits,
    int N, int T, int B,
    float rms_eps,
    int num_sm,
    cudaStream_t stream)
{
    constexpr size_t smem_size =
        ((size_t)CHUNK_DEPTH * NC * H * sizeof(bf16_t) + sizeof(FwdSmemPlan<NC>) + 15) &
        ~size_t(15);
    auto kernel = &attn_res_fwd_online_v2_kernel<H, NC, RELEASE_TMEM>;
    static bool attrs_set = false;
    if (!attrs_set) {
        if (smem_size > 48 * 1024) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        attrs_set = true;
    }
    int grid = RELEASE_TMEM ? num_sm * 2 : num_sm;
    kernel<<<grid, BLK, smem_size, stream>>>(
        block_residual, layer_residual, res_weight, rms_weight,
        output, rsigma, probs, logits, N, T, B, rms_eps);
}

template <int H, int TB_TILE>
static void launch_n1_ttile(
    const bf16_t* layer_residual,
    const bf16_t* res_weight,
    const bf16_t* rms_weight,
    bf16_t* output,
    float* rsigma,
    float* probs,
    float* logits,
    int T, int B,
    float rms_eps,
    int num_sm,
    cudaStream_t stream)
{
    constexpr size_t smem_size =
        ((size_t)CHUNK_DEPTH * TB_TILE * H * sizeof(bf16_t) +
         sizeof(FwdSmemPlan<1>) + 15) & ~size_t(15);
    auto kernel = &attn_res_fwd_n1_ttile_kernel<H, TB_TILE, true>;
    static bool attrs_set = false;
    if (!attrs_set) {
        if (smem_size > 48 * 1024) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        attrs_set = true;
    }
    kernel<<<num_sm * 2, BLK, smem_size, stream>>>(
        layer_residual, res_weight, rms_weight,
        output, rsigma, probs, logits, T, B, rms_eps);
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
    cudaGetDevice(&dev);
    int num_sm = attn_res_fwd_grid_size(dev);
    if (num_sm <= 0 || N > N_MAX)
    {
        return;
    }

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
        launch_fwd<7168, 4, false>(block_residual, layer_residual, res_weight, rms_weight, output, rsigma, probs,
            logits, N, T, B, rms_eps, num_sm, stream);
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
}

} // namespace kernels::kimiK3AttnRes

TRTLLM_NAMESPACE_END
