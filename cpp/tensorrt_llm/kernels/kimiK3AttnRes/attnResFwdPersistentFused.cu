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

// Persistent attention-residual kernel WITH the fused epilogue, ported from the
// vLLM copy of this same kernel (vllm/csrc/libtorch_stable/kimi_k3/
// attn_res_kernel.cu, also NVIDIA-copyright). Both descend from one source:
// the constants here (K_TILE, BLK, CHUNK_DEPTH, CONSUMER_GROUPS) and the
// launcher's `grid = RELEASE_TMEM ? num_sm * 2 : num_sm` are identical to
// attnResFwd.cu's attn_res_fwd_online_v2_kernel. What that copy grew and ours
// did not is the epilogue: HAS_DELTA folds the residual add in, and
// HAS_OUTPUT_NORM folds the trailing RMSNorm in.
//
// Why this is worth trying even though a July 2026 GB300 experiment measured a
// persistent-grid epilogue as 41-108% SLOWER at T=8192 (run record
// kimi_k3_attn_res_fusion_kernel_opt/20260730_185026): that experiment's
// epilogue cost ~150 us against a ~36 us standalone RMSNorm, i.e. ~3.9 us of
// added time per token per CTA, which a barrier plus a shuffle tree cannot
// account for -- it must have re-walked H. This version does not: the sum of
// squares is accumulated in the output pass that already exists, acc32 never
// leaves registers, and the only added synchronization is one
// CONSUMER_THREADS-wide named barrier per token. That is a different cost
// structure, so the old measurement does not settle it.
//
// One divergence is a bug fix rather than an adaptation, and it is the reason
// this kernel was unusable before: the source copy triggers the programmatic
// launch completion before barriering, so in a warp-specialized kernel the
// producer warp releases the dependent grid while the consumer warps are still
// storing the output. That is latent upstream, where nothing downstream is
// launched with the attribute, and fatal here, where PDL is on stack-wide by
// default. See the barrier near the end of the kernel.
//
// Deliberately a separate translation unit rather than an edit to
// attnResFwd.cu: everything below sits in an anonymous namespace, so the
// duplicate sm100::fwd_prod_v2 names have internal linkage and do not collide
// with the unfused copy. Keeping both lets one A/B the other on identical
// inputs. If this wins, the follow-up is to merge them, not to keep two.
//
// The __global__ is renamed attn_res_fwd_persistent_fused_kernel rather than
// keeping the source's attn_res_fwd_online_v2_kernel. Internal linkage settles
// the linker but not the profiler: nsys keys on the demangled short name, so
// two same-named kernels in one process collapse into one row and an A/B
// against attnResFwd.cu cannot tell which copy ran. That cost a full
// misattribution pass on the 20260902-094152 GB300 traces -- baseline and
// candidate both showed "attn_res_fwd_online_v2_kernel" and the only thing
// separating them was the launch grid, which is 152 for both whenever
// RELEASE_TMEM is false. Distinct names, so the trace answers it directly.

#include "attnResFwdPersistentFused.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <limits>
#include <mutex>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimiK3AttnRes
{
namespace
{

using bf16_t = __nv_bfloat16;

namespace sm100
{
namespace fwd_prod_v2
{

constexpr int K_TILE = 1024;
constexpr int N_CHUNK_DEFAULT = 4;
constexpr int CHUNK_DEPTH = 2;
constexpr int BLK = 288;                   // 1 producer warp + 8 consumer warps
constexpr int CONSUMER_THREADS = BLK - 32; // 256
constexpr int CONSUMER_WARPS = CONSUMER_THREADS / 32;
constexpr int CONSUMER_GROUPS = 2;         // two 128-thread consumer groups
constexpr int CONSUMER_THREADS_PER_GROUP = CONSUMER_THREADS / CONSUMER_GROUPS;
constexpr int FIRST_USER_NAMED_BARRIER = 8;

__device__ __forceinline__ bf16_t const* residual_addr(bf16_t const* block_res, bf16_t const* layer_res, int source,
    int N, int token, int block_stride_m, int block_stride_r, int H)
{
    if (source < N - 1)
    {
        return block_res + static_cast<long long>(token) * block_stride_m + source * block_stride_r;
    }
    return layer_res + static_cast<long long>(token) * H;
}

__device__ __forceinline__ uint32_t elect_one_sync()
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
        : "r"(0xffffffff));
    return pred;
}

__device__ __forceinline__ void mbarrier_init(uint64_t& barrier, int thread_count)
{
    uint32_t const barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(barrier_addr), "r"(thread_count));
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t& barrier, uint32_t bytes)
{
    uint32_t const barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier_addr), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t& barrier, int phase)
{
    uint32_t const barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra DONE;\n"
        "bra WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(barrier_addr),
        "r"(phase)
        : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t& barrier)
{
    uint32_t const barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    asm volatile(
        "{\n"
        ".reg .b64 state;\n"
        "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
        "}\n" ::"r"(barrier_addr)
        : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void named_barrier_sync(uint32_t num_threads, uint32_t user_barrier_id)
{
    asm volatile("bar.sync %0, %1;" ::"r"(user_barrier_id + FIRST_USER_NAMED_BARRIER), "r"(num_threads) : "memory");
}

__device__ __forceinline__ void tmem_allocate(int num_columns, uint32_t* dst)
{
    uint32_t const dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_addr), "r"(num_columns));
}

__device__ __forceinline__ void tmem_free(uint32_t tmem_ptr, int num_columns)
{
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_ptr), "r"(num_columns));
}

__device__ __forceinline__ void tmem_release_allocation_lock()
{
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

__device__ __forceinline__ void tmem_store_wait()
{
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// The load-side counterpart. tcgen05.ld is asynchronous exactly as tcgen05.st
// is: the destination registers are not readable, and the source columns are
// not reusable, until this has retired. The source copy waits on the store side
// and never on the load side, so pass B consumes f_cache while its loads may
// still be in flight and the next chunk's pass A overwrites the very columns
// pass B is reading -- my_v_tmem + (si * N_CHUNK + n) * VEC carries no chunk
// index, so every chunk lands on the same columns.
//
// Whether that window is lost or not is a matter of scheduling, which is what
// the measurement shows: at M=8192 with an addend, N=7 gave 209, 166 and 224
// rows over 1% row-relative error on three runs of identical inputs, and N=8
// gave 168, 174 and 176. A varying count on fixed inputs is a race, not
// arithmetic.
__device__ __forceinline__ void tmem_load_wait()
{
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

template <int N, typename T>
__device__ __forceinline__ void tmem_load(uint32_t src_addr, T* dst)
{
    uint32_t* values = reinterpret_cast<uint32_t*>(dst);
    if constexpr (N == 8)
    {
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32"
            "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
            : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]), "=r"(values[4]), "=r"(values[5]),
            "=r"(values[6]), "=r"(values[7])
            : "r"(src_addr));
    }
    else
    {
        static_assert(N == 4, "AttnRes TMEM helpers support x4 and x8");
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x4.b32"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
            : "r"(src_addr));
    }
}

template <int N, typename T>
__device__ __forceinline__ void tmem_store(uint32_t dst_addr, T* src)
{
    uint32_t* values = reinterpret_cast<uint32_t*>(src);
    if constexpr (N == 8)
    {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x8.b32"
            "[%8], {%0, %1, %2, %3, %4, %5, %6, %7};\n" ::"r"(values[0]),
            "r"(values[1]), "r"(values[2]), "r"(values[3]), "r"(values[4]), "r"(values[5]), "r"(values[6]),
            "r"(values[7]), "r"(dst_addr));
    }
    else
    {
        static_assert(N == 4, "AttnRes TMEM helpers support x4 and x8");
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x4.b32"
            "[%4], {%0, %1, %2, %3};\n" ::"r"(values[0]),
            "r"(values[1]), "r"(values[2]), "r"(values[3]), "r"(dst_addr));
    }
}

__device__ __forceinline__ float2 float2_add(float2 const& a, float2 const& b)
{
    float2 result;
    asm volatile("add.rn.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return result;
}

__device__ __forceinline__ float2 float2_mul(float2 const& a, float2 const& b)
{
    float2 result;
    asm volatile("mul.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return result;
}

__device__ __forceinline__ float2 float2_fma(float2 const& a, float2 const& b, float2 const& c)
{
    float2 result;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
    return result;
}

template <int NC>
struct FwdSmemPlan
{
    alignas(16) uint64_t bar_ready[CHUNK_DEPTH];
    alignas(16) uint64_t bar_consumed[CHUNK_DEPTH];
    alignas(16) uint64_t bar_output_norm_ready;
    alignas(16) float2 ws_stats[CONSUMER_WARPS][NC];
    uint32_t tmem_base;
};

__device__ __forceinline__ void cp_async_bulk(void* smem_dst, void const* gmem_src, int bytes, uint64_t& mbar)
{
    uint32_t const s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t const m = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], "
        "[%1], %2, [%3];\n" ::"r"(s),
        "l"(gmem_src), "r"(bytes), "r"(m)
        : "memory");
}

template <int H, int N, int NC = N_CHUNK_DEFAULT, int B = 1, bool RELEASE_TMEM = false, bool HAS_DELTA = false,
    bool HAS_OUTPUT_NORM = false, bool OUTPUT_NORM_IN_SMEM = false>
__global__ void __launch_bounds__(BLK, 1)
    attn_res_fwd_persistent_fused_kernel(bf16_t const* __restrict__ block_res, bf16_t const* __restrict__ layer_res,
        // Diverges from the source copy, which folded the residual add in place on
        // layer_res. TensorRT-LLM's attn_res_add_rmsnorm_fwd returns the updated
        // prefix sum as a new tensor. Note that recovering the in-place form by
        // passing layer_res here is NOT valid: both pointers are __restrict__, so
        // the compiler may assume the read and the store do not overlap. That the
        // consumed value comes from the TMA'd shared-memory copy rather than a
        // reload of layer_res removes one hazard, not the no-alias contract.
        bf16_t* __restrict__ updated_layer_res, bf16_t const* __restrict__ delta, bf16_t const* __restrict__ res_w,
        bf16_t const* __restrict__ rms_w, bf16_t* __restrict__ output, int T, int block_stride_m, int block_stride_r,
        float rms_eps, bf16_t const* __restrict__ output_norm_weight, float output_norm_eps)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
    constexpr float LOG2_E = 1.4426950408889634f;
    constexpr int N_CHUNK = NC;
    // The two-source specialization only consumes half of the TMEM columns.
    constexpr int TMEM_COLS_ALLOC = NC == 2 ? 128 : 256;
    constexpr int NUM_BUFS = CHUNK_DEPTH * NC;
    constexpr int NHT = H / K_TILE;
    constexpr int SLICES_PER_GROUP = (NHT + CONSUMER_GROUPS - 1) / CONSUMER_GROUPS;
    constexpr int VEC = 8;
    constexpr int ACC_PER_THREAD = H == 7168 ? 28 : SLICES_PER_GROUP * VEC;
    constexpr int TMEM_V_COLS_PER_GROUP = SLICES_PER_GROUP * N_CHUNK * VEC;
    constexpr int TMEM_V_COLS_TOTAL = CONSUMER_GROUPS * TMEM_V_COLS_PER_GROUP;
    static_assert(TMEM_V_COLS_TOTAL <= TMEM_COLS_ALLOC);
    static_assert(H >= 4096 && H <= 8192);
    static_assert(H % K_TILE == 0);

    int const tid = threadIdx.x;
    int const wid = tid >> 5;
    int const lane = tid & 31;
    int const TB = T * B;
    int const num_ctas = gridDim.x;
    constexpr int num_chunks = (N + N_CHUNK - 1) / N_CHUNK;

    int const comp_wid = wid - 1;
    int const comp_tid = tid - 32;
    int const group = (comp_wid >= 4) ? 1 : 0;
    int const ct_in_group = (comp_tid >= 0) ? (comp_tid & (CONSUMER_THREADS_PER_GROUP - 1)) : -1;
    int const k_local = ct_in_group * VEC;

    constexpr size_t V_BYTES = (size_t) NUM_BUFS * H * sizeof(bf16_t);
    constexpr size_t DELTA_BYTES = HAS_DELTA ? (size_t) CHUNK_DEPTH * H * sizeof(bf16_t) : 0;
    constexpr size_t OUTPUT_NORM_BYTES = OUTPUT_NORM_IN_SMEM ? (size_t) H * sizeof(bf16_t) : 0;
    extern __shared__ __align__(16) char smem_raw[];
    bf16_t* v_bufs = reinterpret_cast<bf16_t*>(smem_raw); // [NUM_BUFS][H]
    bf16_t* delta_bufs = reinterpret_cast<bf16_t*>(smem_raw + V_BYTES);
    bf16_t* output_norm_buf = reinterpret_cast<bf16_t*>(smem_raw + V_BYTES + DELTA_BYTES);
    FwdSmemPlan<NC>& plan = *reinterpret_cast<FwdSmemPlan<NC>*>(smem_raw + V_BYTES + DELTA_BYTES + OUTPUT_NORM_BYTES);

    auto slot_of = [](long long gci, int n) { return (int) (gci % CHUNK_DEPTH) * N_CHUNK + n; };
    auto phase_of = [](long long gci) { return (int) ((gci / CHUNK_DEPTH) & 1); };
    auto buf_ptr = [&](int slot) -> bf16_t* { return v_bufs + slot * H; };
    auto delta_buf_ptr = [&](int chunk_slot) -> bf16_t* { return delta_bufs + chunk_slot * H; };

    if (wid == 0 && elect_one_sync())
    {
#pragma unroll
        for (int i = 0; i < CHUNK_DEPTH; i++)
        {
            mbarrier_init(plan.bar_ready[i], 1);
            mbarrier_init(plan.bar_consumed[i], CONSUMER_WARPS);
        }
        if constexpr (OUTPUT_NORM_IN_SMEM)
        {
            mbarrier_init(plan.bar_output_norm_ready, 1);
        }
        fence_mbarrier_init();
    }

    // gdc wait BEFORE tmem alloc. Unconditional, as in the source copy.
    //
    // A revision of this file gated it on the launch attribute, on the reasoning
    // that attnResFwd.cu's s1 kernels use `if constexpr (ENABLE_PDL)` and that
    // the call is documented as a no-op without a programmatic launch. Measured:
    // removing it collapses MoE routing exactly the way a premature trigger does
    // -- 92.65% of token-to-expert selections on slots 0..15 with the attribute
    // OFF, against 1.87% for the same build with it present. So it is doing
    // something this kernel depends on beyond waiting for a predecessor, and it
    // stays. The s1 kernels are not a precedent: they are not warp-specialized
    // and do not allocate TMEM here.
    cudaGridDependencySynchronize();

    if (wid == 1)
    {
        tmem_allocate(TMEM_COLS_ALLOC, &plan.tmem_base);
        if constexpr (RELEASE_TMEM)
        {
            tmem_release_allocation_lock();
        }
    }
    __syncthreads();

    if constexpr (OUTPUT_NORM_IN_SMEM)
    {
        if (wid == 0 && elect_one_sync())
        {
            mbarrier_expect_tx(plan.bar_output_norm_ready, H * (int) sizeof(bf16_t));
            cp_async_bulk(output_norm_buf, output_norm_weight, H * sizeof(bf16_t), plan.bar_output_norm_ready);
        }
    }

    const uint32_t my_v_tmem = comp_tid >= 0 ? plan.tmem_base + group * TMEM_V_COLS_PER_GROUP : 0;
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
                    int2 rms_v = *reinterpret_cast<int2 const*>(rms_w + h_base);
                    int2 res_v = *reinterpret_cast<int2 const*>(res_w + h_base);
                    auto* rms2 = reinterpret_cast<__nv_bfloat162*>(&rms_v);
                    auto* res2 = reinterpret_cast<__nv_bfloat162*>(&res_v);
#pragma unroll
                    for (int k = 0; k < 2; k++)
                    {
                        float2 rf = __bfloat1622float2(rms2[k]);
                        float2 sf = __bfloat1622float2(res2[k]);
                        q_cache[si * VEC + 2 * k] = rf.x * sf.x;
                        q_cache[si * VEC + 2 * k + 1] = rf.y * sf.y;
                    }
                    continue;
                }
            }
            int dt = si * CONSUMER_GROUPS + group;
            if (dt >= NHT)
                continue;
            int h_base = dt * K_TILE + k_local;
            int4 rms_v = *reinterpret_cast<int4 const*>(rms_w + h_base);
            int4 res_v = *reinterpret_cast<int4 const*>(res_w + h_base);
            auto* rms2 = reinterpret_cast<__nv_bfloat162*>(&rms_v);
            auto* res2 = reinterpret_cast<__nv_bfloat162*>(&res_v);
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                float2 rf = __bfloat1622float2(rms2[k]);
                float2 sf = __bfloat1622float2(res2[k]);
                q_cache[si * VEC + 2 * k] = rf.x * sf.x;
                q_cache[si * VEC + 2 * k + 1] = rf.y * sf.y;
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
                int const t = tb / B;
                for (int ci = 0; ci < num_chunks; ci++, gci++)
                {
                    int ns = ci * N_CHUNK;
                    int an = min(N_CHUNK, N - ns);
                    int chunk_slot = (int) (gci % CHUNK_DEPTH);
                    int pc = phase_of(gci);
                    mbarrier_wait(plan.bar_consumed[chunk_slot], pc ^ 1);
                    int transaction_bytes = an * H * (int) sizeof(bf16_t);
                    if constexpr (HAS_DELTA)
                    {
                        int prefix_n = N - 1 - ns;
                        if (prefix_n >= 0 && prefix_n < an)
                        {
                            transaction_bytes += H * (int) sizeof(bf16_t);
                        }
                    }
                    mbarrier_expect_tx(plan.bar_ready[chunk_slot], transaction_bytes);
#pragma unroll
                    for (int n = 0; n < N_CHUNK; n++)
                    {
                        if (n >= an)
                            continue;
                        int slot = slot_of(gci, n);
                        bf16_t const* src
                            = residual_addr(block_res, layer_res, ns + n, N, t, block_stride_m, block_stride_r, H);
                        cp_async_bulk(buf_ptr(slot), src, H * sizeof(bf16_t), plan.bar_ready[chunk_slot]);
                    }
                    if constexpr (HAS_DELTA)
                    {
                        int prefix_n = N - 1 - ns;
                        if (prefix_n >= 0 && prefix_n < an)
                        {
                            cp_async_bulk(delta_buf_ptr(chunk_slot), delta + (long long) tb * H, H * sizeof(bf16_t),
                                plan.bar_ready[chunk_slot]);
                        }
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

#pragma unroll
            for (int ci = 0; ci < num_chunks; ci++, gci++)
            {
                int ns = ci * N_CHUNK;
                int an = min(N_CHUNK, N - ns);
                int chunk_slot = (int) (gci % CHUNK_DEPTH);
                int pr = phase_of(gci);
                mbarrier_wait(plan.bar_ready[chunk_slot], pr);

                float2 sq_local[N_CHUNK] = {};
                float2 dot_local[N_CHUNK] = {};

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
                                    auto* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
                                    if constexpr (HAS_DELTA)
                                    {
                                        int prefix_n = N - 1 - ns;
                                        if (n == prefix_n)
                                        {
                                            const bf16_t* delta_ptr = delta_buf_ptr(chunk_slot) + h_base;
#pragma unroll
                                            for (int j = 0; j < 2; j++)
                                            {
                                                auto delta2
                                                    = *reinterpret_cast<const __nv_bfloat162*>(delta_ptr + 2 * j);
                                                v2[j] = __hadd2(v2[j], delta2);
                                            }
                                            *reinterpret_cast<int2*>(updated_layer_res + (long long) tb * H + h_base)
                                                = vp;
                                        }
                                    }
                                    float2 f[2] = {__bfloat1622float2(v2[0]), __bfloat1622float2(v2[1])};
                                    tmem_store<4>(my_v_tmem + (si * N_CHUNK + n) * VEC, f);
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
                        int h_base = dt * K_TILE + k_local;
                        const float* qv = &q_cache[si * VEC];

#pragma unroll
                        for (int n = 0; n < AN; n++)
                        {
                            int slot = slot_of(gci, n);
                            int4 vp = *reinterpret_cast<const int4*>(buf_ptr(slot) + h_base);
                            auto* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
                            if constexpr (HAS_DELTA)
                            {
                                int prefix_n = N - 1 - ns;
                                if (n == prefix_n)
                                {
                                    const bf16_t* delta_ptr = delta_buf_ptr(chunk_slot) + h_base;
#pragma unroll
                                    for (int j = 0; j < VEC / 2; j++)
                                    {
                                        auto delta2 = *reinterpret_cast<const __nv_bfloat162*>(delta_ptr + 2 * j);
                                        v2[j] = __hadd2(v2[j], delta2);
                                    }
                                    *reinterpret_cast<int4*>(updated_layer_res + (long long) tb * H + h_base) = vp;
                                }
                            }
                            float2 f[4] = {__bfloat1622float2(v2[0]), __bfloat1622float2(v2[1]),
                                __bfloat1622float2(v2[2]), __bfloat1622float2(v2[3])};
                            tmem_store<VEC>(my_v_tmem + (si * N_CHUNK + n) * VEC, f);
#pragma unroll
                            for (int j = 0; j < VEC / 2; j++)
                            {
                                sq_local[n] = float2_fma(f[j], f[j], sq_local[n]);
                                dot_local[n] = float2_fma(f[j], make_float2(qv[2 * j], qv[2 * j + 1]), dot_local[n]);
                            }
                        }
                    }
                };
                if constexpr (NC == 4)
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
                if (lane == 0)
                {
                    mbarrier_arrive(plan.bar_consumed[chunk_slot]);
                }
                tmem_store_wait();

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
                if (lane == 0)
                {
#pragma unroll
                    for (int n = 0; n < N_CHUNK; n++)
                    {
                        plan.ws_stats[comp_wid][n] = reduce_pair[n];
                    }
                }
                named_barrier_sync(CONSUMER_THREADS, 0);

                float local_rsig = 0.f;
                float local_logit = 0.f;
                int stat_n = lane / CONSUMER_WARPS;
                int stat_w = lane % CONSUMER_WARPS;
                float2 totals = {};
                if (stat_n < N_CHUNK)
                {
                    totals = plan.ws_stats[stat_w][stat_n];
                }
                // Second barrier: ws_stats is a single buffer reused every chunk, and
                // the barrier above only orders the writes before the reads. Without
                // this one a warp that has finished reading runs ahead into the next
                // chunk and writes its own row while a slower warp is still reading
                // that row for the current chunk -- a write-after-read on shared
                // memory that no amount of correct barrier usage upstream prevents.
                //
                // Only reachable when a token spans more than one chunk, which is
                // exactly the condition every failing shape shares.
                named_barrier_sync(CONSUMER_THREADS, 0);
#pragma unroll
                for (int offset = CONSUMER_WARPS / 2; offset > 0; offset >>= 1)
                {
                    totals.x += __shfl_down_sync(0xffffffff, totals.x, offset, CONSUMER_WARPS);
                    totals.y += __shfl_down_sync(0xffffffff, totals.y, offset, CONSUMER_WARPS);
                }
                if (stat_n < N_CHUNK && stat_w == 0)
                {
                    local_rsig = rsqrtf(totals.x / H + eps_cache);
                    local_logit = totals.y * local_rsig;
                }
                float logit_n[N_CHUNK];
#pragma unroll
                for (int n = 0; n < N_CHUNK; n++)
                {
                    logit_n[n] = __shfl_sync(0xffffffff, local_logit, n * CONSUMER_WARPS);
                }

                float m_chunk = -FLT_MAX;
#pragma unroll
                for (int n = 0; n < N_CHUNK; n++)
                {
                    if (n < an)
                        m_chunk = fmaxf(m_chunk, logit_n[n]);
                }
                float m_new = fmaxf(m_running, m_chunk);
                float corr = exp2f((m_running - m_new) * LOG2_E);
                float w_n[N_CHUNK] = {};
                float w_sum = 0.f;
#pragma unroll
                for (int n = 0; n < N_CHUNK; n++)
                {
                    if (n < an)
                    {
                        w_n[n] = exp2f((logit_n[n] - m_new) * LOG2_E);
                        w_sum += w_n[n];
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
                                    tmem_load<4>(my_v_tmem + (si * N_CHUNK + n) * VEC, f_cache[n]);
                                }
                                tmem_load_wait();
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
                        float2 corr2 = make_float2(corr, corr);
                        float2 a[VEC / 2];
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
                            tmem_load<VEC>(my_v_tmem + (si * N_CHUNK + n) * VEC, f_cache[n]);
                        }
                        tmem_load_wait();
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
                if constexpr (NC == 4)
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
            }

            float inv_s = 1.f / s_running;
            bf16_t* out_ptr = output + (long long) tb * H;
            float2 output_sq_pair = {};
            // When output RMSNorm is fused, the softmax denominator cancels:
            // (acc / s) * rsqrt(mean((acc / s)^2) + eps)
            //   = acc * rsqrt(mean(acc^2) + eps * s^2).
#pragma unroll
            for (int si = 0; si < SLICES_PER_GROUP; si++)
            {
                if constexpr (H == 7168)
                {
                    if (si == SLICES_PER_GROUP - 1)
                    {
                        int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
                        uint2 packed;
                        auto* ov2 = reinterpret_cast<__nv_bfloat162*>(&packed);
                        float2 inv2 = make_float2(inv_s, inv_s);
#pragma unroll
                        for (int j = 0; j < 2; j++)
                        {
                            float2 old = make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
                            if constexpr (HAS_OUTPUT_NORM)
                            {
                                output_sq_pair = float2_fma(old, old, output_sq_pair);
                            }
                            else
                            {
                                float2 mixed = float2_mul(old, inv2);
                                ov2[j] = __float22bfloat162_rn(mixed);
                            }
                        }
                        if constexpr (!HAS_OUTPUT_NORM)
                        {
                            *reinterpret_cast<uint2*>(out_ptr + h_base) = packed;
                        }
                        continue;
                    }
                }
                int dt = si * CONSUMER_GROUPS + group;
                if (dt >= NHT)
                    continue;
                int h_base = dt * K_TILE + k_local;
                uint4 packed;
                auto* ov2 = reinterpret_cast<__nv_bfloat162*>(&packed);
                float2 inv2 = make_float2(inv_s, inv_s);
#pragma unroll
                for (int j = 0; j < VEC / 2; j++)
                {
                    float2 old = make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
                    if constexpr (HAS_OUTPUT_NORM)
                    {
                        output_sq_pair = float2_fma(old, old, output_sq_pair);
                    }
                    else
                    {
                        float2 mixed = float2_mul(old, inv2);
                        ov2[j] = __float22bfloat162_rn(mixed);
                    }
                }
                if constexpr (!HAS_OUTPUT_NORM)
                {
                    *reinterpret_cast<uint4*>(out_ptr + h_base) = packed;
                }
            }

            if constexpr (HAS_OUTPUT_NORM)
            {
                if constexpr (OUTPUT_NORM_IN_SMEM)
                {
                    // The immutable weight copy is acquired once, at its first use.
                    if (tb == blockIdx.x)
                    {
                        mbarrier_wait(plan.bar_output_norm_ready, 0);
                    }
                }
                float output_sq = output_sq_pair.x + output_sq_pair.y;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                {
                    output_sq += __shfl_xor_sync(0xffffffff, output_sq, offset);
                }
                if (lane == 0)
                {
                    plan.ws_stats[comp_wid][0] = make_float2(output_sq, 0.f);
                }
                named_barrier_sync(CONSUMER_THREADS, 0);
                float total_sq = lane < CONSUMER_WARPS ? plan.ws_stats[lane][0].x : 0.f;
                // Same write-after-read as in the chunk loop, across the token
                // boundary: the next token's first chunk writes ws_stats again.
                named_barrier_sync(CONSUMER_THREADS, 0);
#pragma unroll
                for (int offset = CONSUMER_WARPS / 2; offset > 0; offset >>= 1)
                {
                    total_sq += __shfl_down_sync(0xffffffff, total_sq, offset, CONSUMER_WARPS);
                }
                if (lane == 0)
                {
                    total_sq = rsqrtf(total_sq / H + output_norm_eps * s_running * s_running);
                }
                float output_rsigma = __shfl_sync(0xffffffff, total_sq, 0);
#pragma unroll
                for (int si = 0; si < SLICES_PER_GROUP; si++)
                {
                    if constexpr (H == 7168)
                    {
                        if (si == SLICES_PER_GROUP - 1)
                        {
                            int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
                            uint2 packed;
                            auto* values = reinterpret_cast<bf16_t*>(&packed);
#pragma unroll
                            for (int j = 0; j < 4; j++)
                            {
                                bf16_t const* weight_ptr = OUTPUT_NORM_IN_SMEM ? output_norm_buf : output_norm_weight;
                                float weight = __bfloat162float(weight_ptr[h_base + j]);
                                values[j] = __float2bfloat16(acc32[si * VEC + j] * output_rsigma * weight);
                            }
                            *reinterpret_cast<uint2*>(out_ptr + h_base) = packed;
                            continue;
                        }
                    }
                    int dt = si * CONSUMER_GROUPS + group;
                    if (dt >= NHT)
                        continue;
                    int h_base = dt * K_TILE + k_local;
                    uint4 packed;
                    auto* values = reinterpret_cast<bf16_t*>(&packed);
#pragma unroll
                    for (int j = 0; j < VEC; j++)
                    {
                        bf16_t const* weight_ptr = OUTPUT_NORM_IN_SMEM ? output_norm_buf : output_norm_weight;
                        float weight = __bfloat162float(weight_ptr[h_base + j]);
                        values[j] = __float2bfloat16(acc32[si * VEC + j] * output_rsigma * weight);
                    }
                    *reinterpret_cast<uint4*>(out_ptr + h_base) = packed;
                }
            }
        }
    }

    // Barrier BEFORE the trigger, not after. This kernel is warp-specialized:
    // the producer warp leaves its loop as soon as the last TMA is issued and
    // never waits for the consumers, so with the source copy's ordering one warp
    // announced "dependent grids may launch" while the other warps of the same
    // CTA were still storing to `output` and `updated_layer_res`. Under PDL that
    // releases the next kernel early onto memory that has not been written yet.
    //
    // The source copy has the same ordering and does not trip on it, because
    // nothing downstream of it is launched with the attribute set, so the trigger
    // has no receiver. TensorRT-LLM enables PDL stack-wide -- getEnvEnablePDL()
    // returns true unless TRTLLM_ENABLE_PDL says otherwise -- so here the next
    // kernel really does start early. Measured: with PDL on, 98.8% of MoE
    // token-to-expert selections landed on slots 0..15, which is what a top-16
    // takes when every expert logit is equal, i.e. when the router read a buffer
    // that was never written. With PDL off the same build reproduces the
    // baseline histogram exactly (run record 0903_persistent_pdl_race.md).
    //
    // attnResFwd.cu's s1 kernels need no barrier here because every thread writes
    // its own slice and then reaches the trigger; only a warp-specialized kernel
    // can arrive at it with writes outstanding.
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
    if (wid == 1)
    {
        tmem_free(plan.tmem_base, TMEM_COLS_ALLOC);
    }
#else
    if (threadIdx.x == 0)
    {
        printf("attn_res_fwd_persistent_fused_kernel requires sm_10x\n");
    }
#endif
}

template <int H, int N, int NC = N_CHUNK_DEFAULT, bool RELEASE_TMEM = false, bool HAS_DELTA = false,
    bool HAS_OUTPUT_NORM = false, bool OUTPUT_NORM_IN_SMEM = false>
static void launch_fwd(bf16_t const* block_residual, bf16_t const* layer_residual, bf16_t* updated_layer_residual,
    bf16_t const* delta, bf16_t const* res_weight, bf16_t const* rms_weight, bf16_t* output, int T, float rms_eps,
    int num_sm, cudaStream_t stream, bf16_t const* output_norm_weight = nullptr, float output_norm_eps = 0.f,
    int block_stride_m = 0, int block_stride_r = 0)
{
    constexpr size_t smem_size
        = ((size_t) CHUNK_DEPTH * (NC + (HAS_DELTA ? 1 : 0)) * H * sizeof(bf16_t)
              + (OUTPUT_NORM_IN_SMEM ? (size_t) H * sizeof(bf16_t) : 0) + sizeof(FwdSmemPlan<NC>) + 15)
        & ~size_t(15);
    auto kernel = &attn_res_fwd_persistent_fused_kernel<H, N, NC, 1, RELEASE_TMEM, HAS_DELTA, HAS_OUTPUT_NORM,
        OUTPUT_NORM_IN_SMEM>;
    // Diverges from the source copy, which used a plain `static bool attrs_set`:
    // that is neither thread-safe nor per-device, so on a multi-GPU process the
    // first caller can leave every other device without the opt-in and the launch
    // fails with an invalid argument. Mirrors attnResFwd.cu's launcher instead.
    if (smem_size > 48 * 1024)
    {
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
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = BLK;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    // Only the attribute is gated; the kernel's two intrinsics stay unconditional
    // as in the source copy. Gating those as well was tried and measured worse:
    // with the attribute off, dropping cudaGridDependencySynchronize() collapsed
    // MoE routing (92.65% of selections on slots 0..15 versus 1.87% with it
    // kept), so it is load-bearing beyond its documented no-op-without-PDL
    // behaviour. Leaving it in costs nothing when no programmatic launch is in
    // play, and the trigger has no receiver in that case either.
    bool const enable_pdl = tensorrt_llm::common::getEnvEnablePDL();
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = enable_pdl ? 1 : 0;
    cudaLaunchKernelEx(&config, kernel, block_residual, layer_residual, updated_layer_residual, delta, res_weight,
        rms_weight, output, T, block_stride_m, block_stride_r, rms_eps, output_norm_weight, output_norm_eps);
}

} // namespace fwd_prod_v2
} // namespace sm100

} // anonymous namespace

bool attnResPersistentFusedSupported(AttnResFwdParams const& params)
{
    // layerResidualAdd is optional: 95 of the ~180 per-forward call sites (the
    // pre-attention mix, and the post-attention mix on snapshot-write layers)
    // have no residual add to fold. Requiring one would have left them on the
    // per-token path and halved whatever this kernel is worth.
    bool const add_ok = params.layerResidualAdd == nullptr || params.updatedLayerResidual != nullptr;
    return params.hiddenSize == 7168 && params.batchSize == 1 && params.numCandidates >= 2 && params.numCandidates <= 9
        && params.blockResidual != nullptr && params.outputRmsWeight != nullptr && add_ok && params.seqLen >= 1;
}

void invokeAttnResPersistentFusedFwd(AttnResFwdParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(attnResPersistentFusedSupported(params),
        "attn_res persistent fused: unsupported shape (H=%d, B=%d, N=%d, T=%d) or a required tensor is null",
        params.hiddenSize, params.batchSize, params.numCandidates, params.seqLen);
    // The per-candidate diagnostics have no place to come from here: this
    // kernel keeps the softmax state in registers across the persistent token
    // loop and never materializes per-candidate rows.
    TLLM_CHECK_WITH_INFO(params.rsigma == nullptr && params.probs == nullptr && params.logits == nullptr,
        "attn_res persistent fused: rsigma/probs/logits are not produced by this kernel");

    int dev = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&dev));
    int num_sm = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev));
    TLLM_CHECK_WITH_INFO(num_sm > 0, "attn_res persistent fused: failed to query the SM count of device %d", dev);

    using namespace sm100::fwd_prod_v2;

    // blockResidual is [N-1, T, B, H] with B == 1, so a token step is H and a
    // candidate step is T * H. The source copy carried a token-major bank and
    // therefore takes both strides as arguments; passing ours keeps the kernel
    // body unchanged.
    int const T = params.seqLen;
    int const block_stride_m = 7168;
    long long const block_stride_r = static_cast<long long>(T) * 7168;
    TLLM_CHECK_WITH_INFO(block_stride_r <= std::numeric_limits<int>::max(),
        "attn_res persistent fused: candidate stride %lld overflows the kernel's int stride", block_stride_r);
    // One candidate step is not the largest address the kernel forms.
    // ``residual_addr`` widens the token term but evaluates
    // ``source * block_stride_r`` in int arithmetic, and ``source`` reaches
    // ``numCandidates - 2`` (the ``source < N - 1`` branch). With N == 9 that
    // product overflows from T == 42801 upward and the load leaves
    // blockResidual. Persistent topology accepts any token count, so nothing
    // upstream bounds T.
    long long const max_candidate_offset = block_stride_r * (params.numCandidates - 2);
    TLLM_CHECK_WITH_INFO(max_candidate_offset <= std::numeric_limits<int>::max(),
        "attn_res persistent fused: candidate offset %lld (T=%d, N=%d) overflows the kernel's int stride",
        max_candidate_offset, T, params.numCandidates);

    bool const has_delta = params.layerResidualAdd != nullptr;
    auto launch = [&](auto nsrc_tok, auto nc_tok, auto release_tmem_tok)
    {
        constexpr int NSRC = decltype(nsrc_tok)::value;
        constexpr int NC = decltype(nc_tok)::value;
        constexpr bool RELEASE_TMEM = decltype(release_tmem_tok)::value;
        auto go = [&](auto has_delta_tok)
        {
            constexpr bool HAS_DELTA = decltype(has_delta_tok)::value;
            launch_fwd<7168, NSRC, NC, RELEASE_TMEM, HAS_DELTA, /*HAS_OUTPUT_NORM=*/true,
                /*OUTPUT_NORM_IN_SMEM=*/true>(params.blockResidual, params.layerResidual, params.updatedLayerResidual,
                params.layerResidualAdd, params.resWeight, params.rmsWeight, params.output, T, params.rmsEps, num_sm,
                stream, params.outputRmsWeight, params.outputRmsEps, block_stride_m, static_cast<int>(block_stride_r));
        };
        if (has_delta)
        {
            go(std::true_type{});
        }
        else
        {
            go(std::false_type{});
        }
    };

    // NC is 3 and RELEASE_TMEM is false for every shape, which is what both
    // ancestors do at H=7168:
    //
    //   vLLM attn_res_kernel.cu:973   launch_fwd<7168, NUM_SOURCES, 3, false, ...>
    //                                 the file's ONLY instantiation, so its own
    //                                 NC==2 branches are dead code there
    //   attnResFwd.cu:1826            launch_fwd<7168, 4, false>
    //
    // This port briefly carried an extra gate selecting NC=2 with RELEASE_TMEM
    // when numCandidates==9 and T>=4096, under a comment claiming it was
    // "inherited from the source copy". It was not: neither ancestor selects
    // NC=2 or RELEASE_TMEM at this hidden size, and the gate fired on exactly
    // one shape -- the deepest prefill layers and nothing else.
    //
    // It is gone rather than kept behind a flag. A four-phase ABBA measured it
    // as no effect: 72921.94 / 72827.29 ms for NC=3 against 72883.91 /
    // 72947.56 for NC=2, a between-arm delta of 0.056% against a within-arm
    // spread of 0.130%, with the four values interleaving. So it buys nothing,
    // and what it costs is a code path upstream never compiles and therefore
    // never exercises. RELEASE_TMEM is not only a tile choice: it doubles
    // occupancy to two CTAs per SM and adds a TMEM dealloc/realloc, which are
    // properties of how the kernel shares the machine rather than of what it
    // computes, and single-GPU measurement cannot see them.
    {
        auto dispatch3 = [&](auto nsrc_tok) { launch(nsrc_tok, std::integral_constant<int, 3>{}, std::false_type{}); };
        switch (params.numCandidates)
        {
        case 2: dispatch3(std::integral_constant<int, 2>{}); break;
        case 3: dispatch3(std::integral_constant<int, 3>{}); break;
        case 4: dispatch3(std::integral_constant<int, 4>{}); break;
        case 5: dispatch3(std::integral_constant<int, 5>{}); break;
        case 6: dispatch3(std::integral_constant<int, 6>{}); break;
        case 7: dispatch3(std::integral_constant<int, 7>{}); break;
        case 8: dispatch3(std::integral_constant<int, 8>{}); break;
        case 9: dispatch3(std::integral_constant<int, 9>{}); break;
        default:
            TLLM_CHECK_WITH_INFO(
                false, "attn_res persistent fused: unsupported N=%d (must be in [2, 9])", params.numCandidates);
        }
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels::kimiK3AttnRes

TRTLLM_NAMESPACE_END
