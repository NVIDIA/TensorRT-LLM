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

#include "tensorrt_llm/kernels/minimaxM3/mcAllreduceNorm.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// The kernels use sm_100-family features (thread-block clusters, cp.async.bulk, st.async, mbarrier transaction
// counts, block-scaled conversions). Every other architecture of the build compiles them to empty bodies; the host
// entry points refuse to launch on those devices.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)                                         \
    && (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)                                     \
        || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
#define MINIMAX_M3_KERNELS_ENABLED 1
#else
#define MINIMAX_M3_KERNELS_ENABLED 0
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace cg = cooperative_groups;

namespace
{

constexpr int kEmpty = static_cast<int>(0x80000000u);
constexpr int kFinMaxTopk = 8; // MoE finalize: top-k rows loaded together up to this k

__device__ __forceinline__ uint32_t canon_bf16x2(uint32_t w)
{
    if ((w & 0xFFFFu) == 0x8000u)
        w &= 0xFFFF0000u;
    if ((w >> 16) == 0x8000u)
        w &= 0x0000FFFFu;
    return w;
}

__device__ __forceinline__ float bf16_lo(uint32_t w)
{
    return __uint_as_float(w << 16);
}

__device__ __forceinline__ float rcp_approx_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ float bf16_hi(uint32_t w)
{
    return __uint_as_float(w & 0xFFFF0000u);
}

__device__ __forceinline__ int4 ld_volatile_v4(int const* p)
{
    int4 v;
    asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

struct FinalizeArgs
{
    __nv_bfloat16 const* fc2; // [permuted rows, H] routed-expert FC2 output (null: plain input x)
    int const* perm;          // [M, topk] expanded index -> permuted row, -1 = dropped
    void const* scales;       // [M, topk] routing weights (fp32 or bf16)
    int scale_fp32;
    int topk;
};

// grid = min(M, MAXT) * CLUSTER CTAs, cluster (CLUSTER, 1, 1), THREADS threads.
// OPTS (each bit-identical): 1 residual + norm weight loaded at entry (before the poll); 2 cluster RMS exchange by
// st.async + mbarrier complete_tx instead of cluster.sync; 4 call parity rotated right after the poll; 8 max-smem
// carveout so the next kernel's CTAs co-reside early.
template <int CLUSTER, int THREADS, int MAXT, int OPTS>
__global__ void __launch_bounds__(THREADS) mc_allreduce_norm_kernel(__nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ x2, FinalizeArgs fin, int M, int H, __nv_bfloat16 const* __restrict__ residual,
    __nv_bfloat16 const* __restrict__ norm_w, float eps, int* __restrict__ ws_uc, int* __restrict__ ws_mc,
    int* __restrict__ flags, int rank, int world, __nv_bfloat16* __restrict__ out_normed,
    __nv_bfloat16* __restrict__ out_residual, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, int sf_k_tiles,
    int prepushed)
{
#if MINIMAX_M3_KERNELS_ENABLED
    __shared__ float warp_sq[THREADS / 32];
    __shared__ alignas(16) float cluster_sq[CLUSTER];
    __shared__ alignas(8) unsigned long long xbar;
    cg::cluster_group cluster = cg::this_cluster();
    if constexpr ((OPTS & 2) != 0)
    {
        if (threadIdx.x == 0)
        {
            uint32_t const b = static_cast<uint32_t>(__cvta_generic_to_shared(&xbar));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b) : "memory");
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    }
    int const crank = static_cast<int>(cluster.block_rank());
    int const t = blockIdx.x / CLUSTER;
    int const cols = H / CLUSTER;
    int const c0 = crank * cols + threadIdx.x * 8;
    bool const active = t < M && threadIdx.x * 8 < cols;
    int64_t const half_words = static_cast<int64_t>(MAXT) * world * (H / 2);

    // prepushed 1: the producer (m3_mx_gemv) already pushed this rank's rows into the exchange slots; 2: and it
    // launched this grid only after its pushes (its grid wait passed), so the slots are polled without a grid wait;
    // 3: as 2, with x in set A and x2 in set B.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (prepushed < 2)
        asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    int const parity = *reinterpret_cast<int volatile*>(flags + blockIdx.x) & 1;
    uint4 rraw = make_uint4(0u, 0u, 0u, 0u);
    uint4 wraw = make_uint4(0u, 0u, 0u, 0u);
    if constexpr ((OPTS & 1) != 0)
    {
        // the residual row was written grids earlier (prepushed >= 2: the producer launched this grid only after its
        // own grid wait; otherwise the grid wait above)
        if (active)
        {
            rraw = *reinterpret_cast<uint4 const*>(residual + static_cast<int64_t>(t) * H + c0);
            wraw = *reinterpret_cast<uint4 const*>(norm_w + c0);
        }
    }
    // Push this rank's slice into every rank's buffer.
    if (active && !prepushed)
    {
        uint4 raw;
        if (fin.fc2 != nullptr)
        {
            // MoE finalize: fp32 sum of the routing-weighted top-k expert rows plus the shared expert, rounded once.
            float f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            if (fin.topk <= kFinMaxTopk)
            {
                // All top-k indices and scales, then every expert-row load in flight at once, then the same in-order
                // FP32 accumulation as the loop below (one memory round trip instead of topk; bit-identical).
                int prow[kFinMaxTopk];
                float sc[kFinMaxTopk];
#pragma unroll
                for (int k = 0; k < kFinMaxTopk; ++k)
                {
                    prow[k] = -1;
                    sc[k] = 0.f;
                    if (k < fin.topk)
                    {
                        int const e = t * fin.topk + k;
                        prow[k] = fin.perm[e];
                        sc[k] = fin.scale_fp32 ? static_cast<float const*>(fin.scales)[e]
                                               : __bfloat162float(static_cast<__nv_bfloat16 const*>(fin.scales)[e]);
                    }
                }
                uint4 rv[kFinMaxTopk];
#pragma unroll
                for (int k = 0; k < kFinMaxTopk; ++k)
                {
                    if (prow[k] >= 0)
                        rv[k] = *reinterpret_cast<uint4 const*>(fin.fc2 + static_cast<int64_t>(prow[k]) * H + c0);
                }
#pragma unroll
                for (int k = 0; k < kFinMaxTopk; ++k)
                {
                    if (prow[k] < 0)
                        continue;
                    __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&rv[k]);
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        float2 const v = __bfloat1622float2(h[q]);
                        f[2 * q] += v.x * sc[k];
                        f[2 * q + 1] += v.y * sc[k];
                    }
                }
            }
            else
                for (int k = 0; k < fin.topk; ++k)
                {
                    int const e = t * fin.topk + k;
                    int const prow = fin.perm[e];
                    if (prow < 0)
                        continue;
                    float const sc = fin.scale_fp32
                        ? static_cast<float const*>(fin.scales)[e]
                        : __bfloat162float(static_cast<__nv_bfloat16 const*>(fin.scales)[e]);
                    uint4 const rv = *reinterpret_cast<uint4 const*>(fin.fc2 + static_cast<int64_t>(prow) * H + c0);
                    __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&rv);
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        float2 const v = __bfloat1622float2(h[q]);
                        f[2 * q] += v.x * sc;
                        f[2 * q + 1] += v.y * sc;
                    }
                }
            if (x2 != nullptr)
            {
                // As stock: the routed output is rounded to BF16 first, then the shared expert is added (one more
                // BF16 rounding of the FP32 sum).
                uint4 const rv = *reinterpret_cast<uint4 const*>(x2 + static_cast<int64_t>(t) * H + c0);
                __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&rv);
#pragma unroll
                for (int q = 0; q < 4; ++q)
                {
                    float2 const routed = __bfloat1622float2(__floats2bfloat162_rn(f[2 * q], f[2 * q + 1]));
                    float2 const v = __bfloat1622float2(h[q]);
                    f[2 * q] = routed.x + v.x;
                    f[2 * q + 1] = routed.y + v.y;
                }
            }
            __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
            for (int q = 0; q < 4; ++q)
                o[q] = __floats2bfloat162_rn(f[2 * q], f[2 * q + 1]);
        }
        else
        {
            raw = *reinterpret_cast<uint4 const*>(x + static_cast<int64_t>(t) * H + c0);
        }
        if (fin.fc2 == nullptr && x2 != nullptr)
        {
            uint4 const raw2 = *reinterpret_cast<uint4 const*>(x2 + static_cast<int64_t>(t) * H + c0);
            __nv_bfloat162* a = reinterpret_cast<__nv_bfloat162*>(&raw);
            __nv_bfloat162 const* b = reinterpret_cast<__nv_bfloat162 const*>(&raw2);
#pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                float2 const fa = __bfloat1622float2(a[q]);
                float2 const fb = __bfloat1622float2(b[q]);
                a[q] = __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
            }
        }
        int64_t const slot = parity * half_words + (static_cast<int64_t>(t) * world + rank) * (H / 2) + c0 / 2;
        *reinterpret_cast<int4*>(ws_mc + slot)
            = make_int4(static_cast<int>(canon_bf16x2(raw.x)), static_cast<int>(canon_bf16x2(raw.y)),
                static_cast<int>(canon_bf16x2(raw.z)), static_cast<int>(canon_bf16x2(raw.w)));
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    float s[8] = {};
    float sq = 0.f;
    if (active)
    {
        // The residual row (written grids earlier) and every rank's slot are loaded together; slots still empty are
        // re-polled. Ranks are then summed in rank order, as before.
        if constexpr ((OPTS & 1) == 0)
            rraw = *reinterpret_cast<uint4 const*>(residual + static_cast<int64_t>(t) * H + c0);
        constexpr int kMaxWorld = 8;
        int* const p0 = ws_uc + parity * half_words + static_cast<int64_t>(t) * world * (H / 2) + c0 / 2;
        int4 v[kMaxWorld];
        int4 v2[kMaxWorld];
        bool const two = prepushed == 3;
        int* const p1 = p0 + 2 * half_words; // set B
        uint32_t pending = (1u << world) - 1u;
        uint32_t pending2 = two ? pending : 0u;
        long long spins = 0;
        while ((pending | pending2) != 0u)
        {
#pragma unroll
            for (int r = 0; r < kMaxWorld; ++r)
            {
                if ((pending >> r) & 1u)
                    v[r] = ld_volatile_v4(p0 + static_cast<int64_t>(r) * (H / 2));
                if ((pending2 >> r) & 1u)
                    v2[r] = ld_volatile_v4(p1 + static_cast<int64_t>(r) * (H / 2));
            }
#pragma unroll
            for (int r = 0; r < kMaxWorld; ++r)
            {
                if (((pending >> r) & 1u) && v[r].x != kEmpty && v[r].y != kEmpty && v[r].z != kEmpty
                    && v[r].w != kEmpty)
                    pending &= ~(1u << r);
                if (((pending2 >> r) & 1u) && v2[r].x != kEmpty && v2[r].y != kEmpty && v2[r].z != kEmpty
                    && v2[r].w != kEmpty)
                    pending2 &= ~(1u << r);
            }
            if (++spins > (1LL << 31))
                __trap();
        }
#pragma unroll
        for (int r = 0; r < kMaxWorld; ++r)
        {
            if (r < world)
            {
                *reinterpret_cast<int4*>(p0 + static_cast<int64_t>(r) * (H / 2))
                    = make_int4(kEmpty, kEmpty, kEmpty, kEmpty);
                uint32_t ww[4] = {static_cast<uint32_t>(v[r].x), static_cast<uint32_t>(v[r].y),
                    static_cast<uint32_t>(v[r].z), static_cast<uint32_t>(v[r].w)};
                if (two)
                {
                    *reinterpret_cast<int4*>(p1 + static_cast<int64_t>(r) * (H / 2))
                        = make_int4(kEmpty, kEmpty, kEmpty, kEmpty);
                    uint32_t const w2[4] = {static_cast<uint32_t>(v2[r].x), static_cast<uint32_t>(v2[r].y),
                        static_cast<uint32_t>(v2[r].z), static_cast<uint32_t>(v2[r].w)};
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        __nv_bfloat162 const u
                            = __floats2bfloat162_rn(bf16_lo(ww[q]) + bf16_lo(w2[q]), bf16_hi(ww[q]) + bf16_hi(w2[q]));
                        ww[q] = *reinterpret_cast<uint32_t const*>(&u);
                    }
                }
#pragma unroll
                for (int q = 0; q < 4; ++q)
                {
                    s[2 * q] += bf16_lo(ww[q]);
                    s[2 * q + 1] += bf16_hi(ww[q]);
                }
            }
        }
        __nv_bfloat162 const* rr = reinterpret_cast<__nv_bfloat162 const*>(&rraw);
        uint32_t upd[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            float2 const rf = __bfloat1622float2(rr[q]);
            __nv_bfloat162 const u = __floats2bfloat162_rn(s[2 * q] + rf.x, s[2 * q + 1] + rf.y);
            upd[q] = *reinterpret_cast<uint32_t const*>(&u);
            float2 const uf = __bfloat1622float2(u);
            s[2 * q] = uf.x;
            s[2 * q + 1] = uf.y;
            sq += uf.x * uf.x + uf.y * uf.y;
        }
        *reinterpret_cast<uint4*>(out_residual + static_cast<int64_t>(t) * H + c0)
            = make_uint4(upd[0], upd[1], upd[2], upd[3]);
    }
    if constexpr ((OPTS & 4) != 0)
    {
        // every CTA (active or not) rotates its call parity once its own poll is done (read only by later grids)
        if (threadIdx.x == 0)
            flags[blockIdx.x] = parity ^ 1;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_xor_sync(0xffffffffu, sq, off);
    if ((threadIdx.x & 31) == 0)
        warp_sq[threadIdx.x / 32] = sq;
    __syncthreads();
    if constexpr ((OPTS & 2) != 0)
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); // peers' xbar initialized (long since)
    if (threadIdx.x == 0)
    {
        float total = 0.f;
#pragma unroll
        for (int w = 0; w < THREADS / 32; ++w)
            total += warp_sq[w];
        if constexpr ((OPTS & 2) == 0)
            for (int c = 0; c < CLUSTER; ++c)
                *cluster.map_shared_rank(&cluster_sq[crank], c) = total;
        else
        {
            uint32_t const b = static_cast<uint32_t>(__cvta_generic_to_shared(&xbar));
            uint32_t const d = static_cast<uint32_t>(__cvta_generic_to_shared(&cluster_sq[crank]));
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(CLUSTER * 4)
                         : "memory");
            for (int c = 0; c < CLUSTER; ++c)
            {
                uint32_t rd, rb;
                asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rd) : "r"(d), "r"(c));
                asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(b), "r"(c));
                asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(rd),
                             "r"(__float_as_uint(total)), "r"(rb)
                             : "memory");
            }
        }
    }
    if constexpr ((OPTS & 2) == 0)
        cluster.sync();
    else
    {
        // every peer's sum landed (st.async complete_tx on this CTA's barrier); no cluster-wide release-arrive
        uint32_t const b = static_cast<uint32_t>(__cvta_generic_to_shared(&xbar));
        uint32_t done = 0;
        while (!done)
            asm volatile(
                "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
                " selp.u32 %0, 1, 0, p;\n}"
                : "=r"(done)
                : "r"(b)
                : "memory");
    }
    if (active)
    {
        float total = 0.f;
#pragma unroll
        for (int c = 0; c < CLUSTER; ++c)
            total += cluster_sq[c];
        float const inv = rsqrtf(total / static_cast<float>(H) + eps);
        if constexpr ((OPTS & 1) == 0)
            wraw = *reinterpret_cast<uint4 const*>(norm_w + c0);
        __nv_bfloat162 const* wv = reinterpret_cast<__nv_bfloat162 const*>(&wraw);
        uint32_t o[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            float2 const wf = __bfloat1622float2(wv[q]);
            __nv_bfloat162 const n = __floats2bfloat162_rn(s[2 * q] * inv * wf.x, s[2 * q + 1] * inv * wf.y);
            o[q] = *reinterpret_cast<uint32_t const*>(&n);
            float2 const nf = __bfloat1622float2(n);
            s[2 * q] = nf.x;
            s[2 * q + 1] = nf.y;
        }
        *reinterpret_cast<uint4*>(out_normed + static_cast<int64_t>(t) * H + c0) = make_uint4(o[0], o[1], o[2], o[3]);
    }
    if (out_q != nullptr)
    {
        // MXFP8 copy of the normed row for the next GEMM, as trtllm::mxfp8_quantize(normed, True): four threads
        // hold one 32-column scale block.
        float amax = 0.f;
        if (active)
        {
#pragma unroll
            for (int e = 0; e < 8; ++e)
                amax = fmaxf(amax, fabsf(s[e]));
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        if (active)
        {
            __nv_fp8_e8m0 code;
            code.__x = __nv_cvt_float_to_e8m0(amax * rcp_approx_ftz(448.0f), __NV_SATFINITE, cudaRoundPosInf);
            float const out_scale = amax != 0.f ? rcp_approx_ftz(static_cast<float>(code)) : 0.0f;
            uint32_t packed[2];
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                __nv_fp8x2_e4m3 const lo(make_float2(s[4 * h] * out_scale, s[4 * h + 1] * out_scale));
                __nv_fp8x2_e4m3 const hi(make_float2(s[4 * h + 2] * out_scale, s[4 * h + 3] * out_scale));
                packed[h] = static_cast<uint32_t>(lo.__x) | (static_cast<uint32_t>(hi.__x) << 16);
            }
            *reinterpret_cast<uint2*>(out_q + static_cast<int64_t>(t) * H + c0) = make_uint2(packed[0], packed[1]);
            if ((threadIdx.x & 3) == 0)
            {
                int const kb = c0 / 32;
                out_sf[static_cast<int64_t>(t / 128) * sf_k_tiles * 512 + (kb / 4) * 512 + (t % 32) * 16
                    + ((t % 128) / 32) * 4 + kb % 4]
                    = code.__x;
            }
        }
    }
    if constexpr ((OPTS & 4) == 0)
    {
        __syncthreads();
        if (threadIdx.x == 0)
            flags[blockIdx.x] = parity ^ 1;
    }
#endif
}

template <int CLUSTER, int THREADS, int MAXT, int OPTS>
void launch(MinimaxM3McAllreduceNormParams const& p, bool pdl, cudaStream_t stream)
{
    int const M = p.numTokens, H = p.hidden;
    TLLM_CHECK_WITH_INFO(H % (CLUSTER * 8) == 0 && H / CLUSTER / 8 <= THREADS, "unsupported hidden size %d", H);
    TLLM_CHECK_WITH_INFO(M >= 1 && M <= MAXT, "%d tokens do not fit the %d-token bucket", M, MAXT);
    auto kernel = mc_allreduce_norm_kernel<CLUSTER, THREADS, MAXT, OPTS>;
    static bool const configured = [&]()
    {
        bool ok = CLUSTER <= 8
            || cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess;
        if ((OPTS & 8) != 0)
            ok = ok
                && cudaFuncSetAttribute(
                       kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared)
                    == cudaSuccess;
        if (!ok)
            cudaGetLastError();
        return ok;
    }();
    TLLM_CHECK_WITH_INFO(configured, "cluster size not supported");
    FinalizeArgs const fin{p.fc2, p.perm, p.scales, p.scaleFp32 ? 1 : 0, p.topk};
    cudaLaunchConfig_t cfg = {};
    // The grid is sized to the call's rows: min(M, MAXT) clusters. Each (row, slice) slot keeps its own parity flag,
    // read by the producers before they push and rotated only by the slot's CTA after it polled: a slot that a call
    // does not use keeps its parity, and every rank runs the same calls, so the parities stay consistent when M
    // changes between calls.
    cfg.gridDim = dim3((M < MAXT ? M : MAXT) * CLUSTER);
    cfg.blockDim = dim3(THREADS);
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = CLUSTER;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    cudaError_t const err
        = cudaLaunchKernelEx(&cfg, kernel, p.x, p.x2, fin, M, H, p.residual, p.normWeight, p.eps, p.wsUc, p.wsMc,
            p.flags, p.rank, p.world, p.outNormed, p.outResidual, p.outQuant, p.outSf, (H / 32 + 3) / 4, p.prepushed);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TLLM_CUDA_CHECK(err);
    }
}

constexpr int kCluster = 8;
constexpr int kThreads = 128;

} // namespace

bool isMinimaxM3McAllreduceNormSupported(int32_t maxTokens, int32_t opts)
{
    return (maxTokens == 8 && (opts == 7 || opts == 15)) || (maxTokens == 16 && opts == 15)
        || (maxTokens == 48 && opts == 15);
}

void invokeMinimaxM3McAllreduceNorm(
    MinimaxM3McAllreduceNormParams const& params, int32_t maxTokens, int32_t opts, bool pdl, cudaStream_t stream)
{
    if (maxTokens == 8 && opts == 7)
    {
        launch<kCluster, kThreads, 8, 7>(params, pdl, stream);
    }
    else if (maxTokens == 8 && opts == 15)
    {
        launch<kCluster, kThreads, 8, 15>(params, pdl, stream);
    }
    else if (maxTokens == 16 && opts == 15)
    {
        launch<kCluster, kThreads, 16, 15>(params, pdl, stream);
    }
    else if (maxTokens == 48 && opts == 15)
    {
        launch<kCluster, kThreads, 48, 15>(params, pdl, stream);
    }
    else
    {
        TLLM_THROW("unsupported (bucket %d, opts %d)", maxTokens, opts);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
