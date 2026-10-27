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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp8/mcAllreduceNormTp8.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// The sm_107 TP8 target's own copy of minimaxM3Sm107Tp2/mcAllreduceNormTp2.cu (the version with the Eagle3 capture
// epilogue, cap_out), verbatim but for the names: the TP8 ext carries it, so the TP8 target no longer needs an image
// whose TP2 op has cap_out. The TP2 file's notes follow unchanged.
//
// SM107 (Rubin) copy of minimaxM3/mcAllreduceNorm.cu for the sm_107 TP8 target. Two changes in the arithmetic: the
// rank sum rounds to BF16 before the residual add, the rounding points of the stock all-reduce + fused residual
// RMSNorm on this target (bf16 all-reduce output, then bf16(residual + sum)); and every call but the MoE finalize runs
// the row kernel below, whose RMS statistic is the stock fused norm's reduction tree (bitwise the stock outputs).
//
// The kernels use sm_100-family features (thread-block clusters, cp.async.bulk, st.async, mbarrier transaction
// counts, block-scaled conversions): they are compiled for every sm_10x target of the build, the sm_100f family target
// included (which runs on SM107). A device pass for any other architecture compiles them to a trap, never to a silent
// empty body, and the host entry point checks that the kernel binary it launches holds sm_10x code (see
// checkBinary).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
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

// The Eagle3 capture written by a capture layer's boundary (cap_out, optional): rmsnorm(updated residual) * fc_norm[i],
// bit for bit the m3_spec_tail capture_norm kernel (kernels/minimaxM3/specTail.cu at H 6144: 256 threads, thread c
// sums the squares of the 8-column chunks c, c + 256 and c + 512 in that order, the warp butterfly 16..1, the 8 warp
// sums added in warp order, rsqrtf(sum / H + eps), bf16(v * rs * w)). Only its layout is reproduced here; its
// arithmetic is the same source.
constexpr int kCapThreads = 256;
constexpr int kCapIters = 3;
constexpr int kCapWarps = kCapThreads / 32;
constexpr int kCapHidden = kCapThreads * kCapIters * 8;

// capture_norm's per-thread chain over one 8-column chunk (continues `ss`).
__device__ __forceinline__ float cap_chunk_sq(uint4 const raw, float ss)
{
    __nv_bfloat16 const* e = reinterpret_cast<__nv_bfloat16 const*>(&raw);
#pragma unroll
    for (int q = 0; q < 8; ++q)
    {
        float const v = __bfloat162float(e[q]);
        ss += v * v;
    }
    return ss;
}

__device__ __forceinline__ float cap_warp_sum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// bf16(v * rs * w) for one 8-column chunk of the updated residual (v as floats) into cap.
__device__ __forceinline__ void cap_store(
    float const (&v)[8], float rs, __nv_bfloat16 const* __restrict__ w, __nv_bfloat16* __restrict__ cap)
{
    uint4 const wr = *reinterpret_cast<uint4 const*>(w);
    __nv_bfloat16 const* we = reinterpret_cast<__nv_bfloat16 const*>(&wr);
    uint4 o;
    __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
    for (int q = 0; q < 8; ++q)
        oe[q] = __float2bfloat16(v[q] * rs * __bfloat162float(we[q]));
    *reinterpret_cast<uint4*>(cap) = o;
}

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

// The MXFP8 publish quantizes exactly as FlashInfer's CuTe DSL mxfp8 quantize (MXFP8QuantizeRunner.CUTE_DSL, MX-TAC's
// pinned decode quantize; quantization_cute_dsl_utils.py), the three helpers below being its PTX verbatim. It equals
// trtllm::mxfp8_quantize bit for bit except on blocks with 0 < amax < 448 * 2^-127 (UE8M0 byte 0), where the
// TRT-LLM kernel's rcp.approx.ftz of the denormal scale gives inf (NaN / garbage E4M3) and this flushes the block to 0.
__device__ __forceinline__ uint32_t mxfp8_ue8m0(float amax)
{
    float value;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(amax), "f"(1.0f / 448.0f));
    uint32_t result;
    asm("{\n"
        " .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
        " .reg .u32 bits, exp_biased, mantissa, bump, r;\n"
        " setp.le.f32 p_zero, %1, 0f00000000;\n"
        " mov.b32 bits, %1;\n"
        " shr.b32 exp_biased, bits, 23;\n"
        " and.b32 exp_biased, exp_biased, 255;\n"
        " and.b32 mantissa, bits, 0x7FFFFF;\n"
        " setp.ne.u32 p_has_mant, mantissa, 0;\n"
        " selp.u32 bump, 1, 0, p_has_mant;\n"
        " setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
        " setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
        " and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
        " @p_tiny_sub mov.u32 bump, 0;\n"
        " add.u32 r, exp_biased, bump;\n"
        " setp.gt.u32 p_ovf, r, 254;\n"
        " selp.u32 r, 254, r, p_ovf;\n"
        " selp.u32 %0, 0, r, p_zero;\n"
        "}\n"
        : "=r"(result)
        : "f"(value));
    return result;
}

__device__ __forceinline__ float mxfp8_inv_scale(uint32_t ue8m0)
{
    float inv;
    asm("{\n"
        " .reg .s32 new_exp;\n"
        " .reg .b32 float_bits;\n"
        " .reg .pred p_zero;\n"
        " setp.eq.u32 p_zero, %1, 0;\n"
        " sub.s32 new_exp, 254, %1;\n"
        " max.s32 new_exp, new_exp, 0;\n"
        " shl.b32 float_bits, new_exp, 23;\n"
        " mov.b32 %0, float_bits;\n"
        " @p_zero mov.b32 %0, 0;\n"
        "}\n"
        : "=f"(inv)
        : "r"(ue8m0));
    return inv;
}

// Two BF16-valued floats (lo, hi) times inv_scale -> two E4M3 bytes (lo in the low byte), saturating to +/-448.
__device__ __forceinline__ uint32_t mxfp8_e4m3x2(float lo, float hi, float inv)
{
    uint16_t pair;
    asm("{\n"
        " .reg .f32 f0, f1;\n"
        " mul.f32 f0, %1, %3;\n"
        " mul.f32 f1, %2, %3;\n"
        " min.f32 f0, f0, 0f43E00000;\n"
        " max.f32 f0, f0, 0fC3E00000;\n"
        " min.f32 f1, f1, 0f43E00000;\n"
        " max.f32 f1, f1, 0fC3E00000;\n"
        " cvt.rn.satfinite.e4m3x2.f32 %0, f1, f0;\n"
        "}\n"
        : "=h"(pair)
        : "f"(lo), "f"(hi), "f"(inv));
    return pair;
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
// OPTS 16 (NOT bit-identical with the others; a different boundary): the Eagle3 drafter's boundary, whose stock is a
// plain BF16 all-reduce, then FlashInfer fused_add_rmsnorm: the RMS statistic and the normalization take the
// unrounded FP32 sum bf16(rank sum) + residual (only the stored residual is rounded to BF16).
// CAP: the Eagle3 capture epilogue (cap_out) is compiled in only where a call passes cap_out, so every other call
// runs the kernel without the capture branches, shared arrays or parameters in its body.
template <int CLUSTER, int THREADS, int MAXT, int OPTS, bool CAP>
__global__ void __launch_bounds__(THREADS) mc_allreduce_norm_kernel(__nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ x2, FinalizeArgs fin, int M, int H, __nv_bfloat16 const* __restrict__ residual,
    __nv_bfloat16 const* __restrict__ norm_w, float eps, int* ws_uc, int* ws_mc, int* __restrict__ flags, int rank,
    int world, __nv_bfloat16* __restrict__ out_normed, __nv_bfloat16* __restrict__ out_residual,
    uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, int sf_k_tiles, int prepushed,
    __nv_bfloat16* __restrict__ cap_out, int64_t cap_stride, __nv_bfloat16 const* __restrict__ cap_w, float cap_eps)
{
#if MINIMAX_M3_KERNELS_ENABLED
    __shared__ float warp_sq[THREADS / 32];
    __shared__ alignas(16) float cluster_sq[CLUSTER];
    extern __shared__ __align__(16) uint4 cap_slice[]; // [cols / 8] this CTA's updated chunks (cap_out only)
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
        // Stock rounding points: the rank sum is a BF16 all-reduce output before the residual add.
#pragma unroll
        for (int i = 0; i < 8; ++i)
            s[i] = __bfloat162float(__float2bfloat16_rn(s[i]));
        __nv_bfloat162 const* rr = reinterpret_cast<__nv_bfloat162 const*>(&rraw);
        uint32_t upd[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            float2 const rf = __bfloat1622float2(rr[q]);
            __nv_bfloat162 const u = __floats2bfloat162_rn(s[2 * q] + rf.x, s[2 * q + 1] + rf.y);
            upd[q] = *reinterpret_cast<uint32_t const*>(&u);
            if constexpr ((OPTS & 16) != 0)
            {
                float const x0 = s[2 * q] + rf.x, x1 = s[2 * q + 1] + rf.y;
                s[2 * q] = x0;
                s[2 * q + 1] = x1;
                sq += x0 * x0 + x1 * x1;
            }
            else
            {
                float2 const uf = __bfloat1622float2(u);
                s[2 * q] = uf.x;
                s[2 * q + 1] = uf.y;
                sq += uf.x * uf.x + uf.y * uf.y;
            }
        }
        *reinterpret_cast<uint4*>(out_residual + static_cast<int64_t>(t) * H + c0)
            = make_uint4(upd[0], upd[1], upd[2], upd[3]);
        if constexpr ((OPTS & 16) == 0)
        {
            if constexpr (CAP)
                cap_slice[threadIdx.x] = make_uint4(upd[0], upd[1], upd[2], upd[3]);
        }
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
    if constexpr ((OPTS & 16) == 0)
    {
        if constexpr (CAP)
        {
            // The row's chunks are spread over the cluster (CTA c holds chunks [c * per, (c + 1) * per)); every CTA
            // forms capture_norm's statistic itself, its 128 threads standing in for capture threads ct and ct + 128.
            __shared__ float cap_red[kCapWarps];
            int const per = cols / 8;
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            float ca = 0.f, cb = 0.f;
#pragma unroll
            for (int it = 0; it < kCapIters; ++it)
            {
                int const ka = it * kCapThreads + static_cast<int>(threadIdx.x);
                int const kb = ka + kCapThreads / 2;
                ca = cap_chunk_sq(*cluster.map_shared_rank(&cap_slice[ka % per], ka / per), ca);
                cb = cap_chunk_sq(*cluster.map_shared_rank(&cap_slice[kb % per], kb / per), cb);
            }
            // Every peer read is done before any CTA moves on (and later exits).
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            ca = cap_warp_sum(ca);
            cb = cap_warp_sum(cb);
            if ((threadIdx.x & 31) == 0)
            {
                cap_red[threadIdx.x / 32] = ca;
                cap_red[threadIdx.x / 32 + kCapWarps / 2] = cb;
            }
            __syncthreads();
            float tot = 0.f;
#pragma unroll
            for (int w = 0; w < kCapWarps; ++w)
                tot += cap_red[w];
            float const rs = rsqrtf(tot / H + cap_eps);
            if (active)
                cap_store(s, rs, cap_w + c0, cap_out + static_cast<int64_t>(t) * cap_stride + c0);
        }
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
        // MXFP8 copy of the normed row for the next GEMM, as the CuTe DSL mxfp8 quantize of `normed` (the tree's
        // decode quantize, MX-TAC; see mxfp8_ue8m0): four threads hold one 32-column scale block.
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
            uint32_t const code = mxfp8_ue8m0(amax);
            float const inv_scale = mxfp8_inv_scale(code);
            uint32_t packed[2];
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                packed[h] = mxfp8_e4m3x2(s[4 * h], s[4 * h + 1], inv_scale)
                    | (mxfp8_e4m3x2(s[4 * h + 2], s[4 * h + 3], inv_scale) << 16);
            }
            *reinterpret_cast<uint2*>(out_q + static_cast<int64_t>(t) * H + c0) = make_uint2(packed[0], packed[1]);
            if ((threadIdx.x & 3) == 0)
            {
                int const kb = c0 / 32;
                out_sf[static_cast<int64_t>(t / 128) * sf_k_tiles * 512 + (kb / 4) * 512 + (t % 32) * 16
                    + ((t % 128) / 32) * 4 + kb % 4]
                    = static_cast<uint8_t>(code);
            }
        }
    }
    if constexpr ((OPTS & 4) == 0)
    {
        __syncthreads();
        if (threadIdx.x == 0)
            flags[blockIdx.x] = parity ^ 1;
    }
#elif defined(__CUDA_ARCH__)
    __trap(); // compiled for an architecture without the sm_10x features: never a silent no-op
#endif
}

// Refuses (raises) to launch a kernel whose loaded binary holds no sm_10x code (PTX and SASS version >= 100): the
// device passes for other architectures compile the kernels to a trap, and a build that left out every sm_10x target
// must not reach the GPU at all.
template <typename Kernel>
bool checkBinary(Kernel kernel, char const* name)
{
    cudaFuncAttributes attr{};
    TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<void const*>(kernel)));
    TLLM_CHECK_WITH_INFO(attr.ptxVersion >= 100 && attr.binaryVersion >= 100,
        "%s: the loaded binary holds no sm_10x code (PTX %d, SASS %d): rebuild with an sm_100f / sm_107a target", name,
        attr.ptxVersion, attr.binaryVersion);
    return true;
}

template <int CLUSTER, int THREADS, int MAXT, int OPTS>
void launch(MinimaxM3Sm107Tp8McAllreduceNormParams const& p, bool pdl, cudaStream_t stream)
{
    int const M = p.numTokens, H = p.hidden;
    TLLM_CHECK_WITH_INFO(H % (CLUSTER * 8) == 0 && H / CLUSTER / 8 <= THREADS, "unsupported hidden size %d", H);
    TLLM_CHECK_WITH_INFO(M >= 1 && M <= MAXT, "%d tokens do not fit the %d-token bucket", M, MAXT);
    auto kernel = p.capOut != nullptr ? mc_allreduce_norm_kernel<CLUSTER, THREADS, MAXT, OPTS, true>
                                      : mc_allreduce_norm_kernel<CLUSTER, THREADS, MAXT, OPTS, false>;
    static bool const binaryOk
        = checkBinary(mc_allreduce_norm_kernel<CLUSTER, THREADS, MAXT, OPTS, false>, "mc_allreduce_norm_kernel")
        && checkBinary(
            mc_allreduce_norm_kernel<CLUSTER, THREADS, MAXT, OPTS, true>, "mc_allreduce_norm_kernel (capture)");
    (void) binaryOk;
    // Both instantiations (with and without the capture epilogue) get the same attributes, whichever runs first.
    static bool const configured = []()
    {
        auto setup = [](auto k)
        {
            bool ok = CLUSTER <= 8
                || cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess;
            if ((OPTS & 8) != 0)
                ok = ok
                    && cudaFuncSetAttribute(
                           k, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared)
                        == cudaSuccess;
            return ok;
        };
        bool const ok = setup(mc_allreduce_norm_kernel<CLUSTER, THREADS, MAXT, OPTS, false>)
            && setup(mc_allreduce_norm_kernel<CLUSTER, THREADS, MAXT, OPTS, true>);
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
    if (p.capOut != nullptr)
    {
        TLLM_CHECK_WITH_INFO((OPTS & 16) == 0 && H == kCapHidden && 2 * THREADS == kCapThreads,
            "the capture epilogue takes H %d on the 128-thread cluster kernel, not the drafter boundary", kCapHidden);
        cfg.dynamicSmemBytes = static_cast<size_t>(H / CLUSTER / 8) * sizeof(uint4);
    }
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = CLUSTER;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    cudaError_t const err = cudaLaunchKernelEx(&cfg, kernel, p.x, p.x2, fin, M, H, p.residual, p.normWeight, p.eps,
        p.wsUc, p.wsMc, p.flags, p.rank, p.world, p.outNormed, p.outResidual, p.outQuant, p.outSf, (H / 32 + 3) / 4,
        p.prepushed, p.capOut, p.capStride, p.capWeight, p.capEps);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TLLM_CUDA_CHECK(err);
    }
}

// ws_uc / ws_mc are this rank's unicast and multicast views of ONE exchange buffer (they alias): no __restrict__.
// prepushed >= 2 contract: the producer (m3_mx_gemv's o_proj push, the m3_moe engine's finalize push) passes its own
// grid dependency, pushes its rows into the exchange, and only then releases this grid (griddepcontrol
// .launch_dependents); this kernel therefore skips griddepcontrol.wait and reads only the pushed (Lamport-polled)
// words, its parity flags and the residual / norm weight, which were written two or more grids earlier.
//
// The TP8 row kernel: one CTA of H / 8 = 768 threads per token row, used for every call but the MoE finalize. Its
// residual add + RMSNorm is TRT-LLM's rms_norm_kernel (customAllReduceKernels.cu), the stock fused norm after the
// all-reduce on this target, operation for operation at H 6144: thread j owns columns [8j, 8j + 8), its sum of squares
// is acc += v * v over them in order, then the warp butterfly 16..1, the 24 warp sums through shared memory into a
// second 32-lane butterfly (lanes >= 24 zero), rsqrtf(acc / H + eps) and (v * denom) * w. With the BF16 rank sum
// before the residual add, the normed rows, the updated residual and the MXFP8 copy are bitwise the stock ones. The
// exchange slots and the per-(row, slice) parity flags keep the cluster kernel's layout (slice = 96 threads), so
// producers that push (the o_proj GEMV) are unchanged.
constexpr int kSlices = 8;
constexpr int kSliceThreads = 96;
constexpr int kRowThreads = 768;

// grid = M CTAs (one per row), THREADS = H / 8 threads. WORLD ranks (compile time). DRAFTER (opts bit 16): the Eagle3
// drafter boundary, whose stock is a plain BF16 all-reduce then FlashInfer fused_add_rmsnorm: the RMS statistic and
// the normalization take the unrounded FP32 sum bf16(rank sum) + residual (only the stored residual is rounded).
// ws_uc / ws_mc alias (one exchange buffer): no __restrict__.
template <int WORLD, int THREADS, bool DRAFTER, bool CAP>
__global__ void __launch_bounds__(THREADS, 1) mc_allreduce_norm_row_kernel(__nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ x2, int H, __nv_bfloat16 const* __restrict__ residual,
    __nv_bfloat16 const* __restrict__ norm_w, float eps, int* ws_uc, int* ws_mc, int* __restrict__ flags, int rank,
    int maxt, __nv_bfloat16* __restrict__ out_normed, __nv_bfloat16* __restrict__ out_residual,
    uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, int sf_k_tiles, int prepushed,
    __nv_bfloat16* __restrict__ cap_out, int64_t cap_stride, __nv_bfloat16 const* __restrict__ cap_w, float cap_eps)
{
#if MINIMAX_M3_KERNELS_ENABLED
    __shared__ float warp_sq[THREADS / 32];
    __shared__ float total_sh;
    extern __shared__ __align__(16) uint4 cap_row[]; // [THREADS] the updated row's chunks (cap_out only)
    int const t = blockIdx.x;
    int const j = threadIdx.x;
    int const c0 = j * 8;
    int const slice = j / kSliceThreads;
    int64_t const half_words = static_cast<int64_t>(maxt) * WORLD * (H / 2);
    if (prepushed < 2)
        asm volatile("griddepcontrol.wait;" ::: "memory");
    int const parity = *reinterpret_cast<int volatile*>(flags + t * kSlices + slice) & 1;
    uint4 const rraw = *reinterpret_cast<uint4 const*>(residual + static_cast<int64_t>(t) * H + c0);
    uint4 const wraw = *reinterpret_cast<uint4 const*>(norm_w + c0);
    if (!prepushed)
    {
        uint4 raw = *reinterpret_cast<uint4 const*>(x + static_cast<int64_t>(t) * H + c0);
        if (x2 != nullptr)
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
        int64_t const slot = parity * half_words + (static_cast<int64_t>(t) * WORLD + rank) * (H / 2) + c0 / 2;
        *reinterpret_cast<int4*>(ws_mc + slot)
            = make_int4(static_cast<int>(canon_bf16x2(raw.x)), static_cast<int>(canon_bf16x2(raw.y)),
                static_cast<int>(canon_bf16x2(raw.z)), static_cast<int>(canon_bf16x2(raw.w)));
    }
    asm volatile("griddepcontrol.launch_dependents;");
    float s[8] = {};
    float sq = 0.f;
    {
        int* const p0 = ws_uc + parity * half_words + static_cast<int64_t>(t) * WORLD * (H / 2) + c0 / 2;
        int* const p1 = p0 + 2 * half_words; // set B
        bool const two = prepushed == 3;
        int4 v[WORLD];
        int4 v2[WORLD];
        uint32_t pending = (1u << WORLD) - 1u;
        uint32_t pending2 = two ? pending : 0u;
        long long spins = 0;
        while ((pending | pending2) != 0u)
        {
#pragma unroll
            for (int r = 0; r < WORLD; ++r)
            {
                if ((pending >> r) & 1u)
                    v[r] = ld_volatile_v4(p0 + static_cast<int64_t>(r) * (H / 2));
                if ((pending2 >> r) & 1u)
                    v2[r] = ld_volatile_v4(p1 + static_cast<int64_t>(r) * (H / 2));
            }
#pragma unroll
            for (int r = 0; r < WORLD; ++r)
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
        for (int r = 0; r < WORLD; ++r)
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
        // Stock rounding points: the rank sum is a BF16 all-reduce output before the residual add.
#pragma unroll
        for (int i = 0; i < 8; ++i)
            s[i] = __bfloat162float(__float2bfloat16_rn(s[i]));
        __nv_bfloat162 const* rr = reinterpret_cast<__nv_bfloat162 const*>(&rraw);
        uint32_t upd[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            float2 const rf = __bfloat1622float2(rr[q]);
            __nv_bfloat162 const u = __floats2bfloat162_rn(s[2 * q] + rf.x, s[2 * q + 1] + rf.y);
            upd[q] = *reinterpret_cast<uint32_t const*>(&u);
            if constexpr (DRAFTER)
            {
                s[2 * q] += rf.x;
                s[2 * q + 1] += rf.y;
            }
            else
            {
                float2 const uf = __bfloat1622float2(u);
                s[2 * q] = uf.x;
                s[2 * q + 1] = uf.y;
            }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e)
            sq += s[e] * s[e];
        *reinterpret_cast<uint4*>(out_residual + static_cast<int64_t>(t) * H + c0)
            = make_uint4(upd[0], upd[1], upd[2], upd[3]);
        if constexpr (!DRAFTER)
        {
            if constexpr (CAP)
                cap_row[j] = make_uint4(upd[0], upd[1], upd[2], upd[3]);
        }
    }
    // TRT-LLM reduce_fusion::block_reduce_sum.
    sq += __shfl_xor_sync(~0, sq, 16);
    sq += __shfl_xor_sync(~0, sq, 8);
    sq += __shfl_xor_sync(~0, sq, 4);
    sq += __shfl_xor_sync(~0, sq, 2);
    sq += __shfl_xor_sync(~0, sq, 1);
    if ((j & 31) == 0)
        warp_sq[j / 32] = sq;
    __syncthreads();
    {
        float val = (j & 31) < THREADS / 32 ? warp_sq[j & 31] : 0.f;
        val += __shfl_xor_sync(~0, val, 16);
        val += __shfl_xor_sync(~0, val, 8);
        val += __shfl_xor_sync(~0, val, 4);
        val += __shfl_xor_sync(~0, val, 2);
        val += __shfl_xor_sync(~0, val, 1);
        if (j == 0)
            total_sh = val;
    }
    // The parity of this row's slices rotates once every thread has read the exchange (later grids only).
    if (j % kSliceThreads == 0)
        flags[t * kSlices + slice] = parity ^ 1;
    __syncthreads();
    if constexpr (!DRAFTER)
    {
        if constexpr (CAP)
        {
            // The whole row is in this CTA: threads 0..255 run capture_norm's chains over the staged chunks.
            __shared__ float cap_red[kCapWarps];
            if (j < kCapThreads)
            {
                float css = 0.f;
#pragma unroll
                for (int it = 0; it < kCapIters; ++it)
                    css = cap_chunk_sq(cap_row[it * kCapThreads + j], css);
                css = cap_warp_sum(css);
                if ((j & 31) == 0)
                    cap_red[j / 32] = css;
            }
            __syncthreads();
            float tot = 0.f;
#pragma unroll
            for (int w = 0; w < kCapWarps; ++w)
                tot += cap_red[w];
            float const rs = rsqrtf(tot / H + cap_eps);
            cap_store(s, rs, cap_w + c0, cap_out + static_cast<int64_t>(t) * cap_stride + c0);
        }
    }
    float const inv = rsqrtf(total_sh / H + eps);
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
    if (out_q != nullptr)
    {
        float amax = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e)
            amax = fmaxf(amax, fabsf(s[e]));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        uint32_t const code = mxfp8_ue8m0(amax);
        float const inv_scale = mxfp8_inv_scale(code);
        uint32_t packed[2];
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            packed[h] = mxfp8_e4m3x2(s[4 * h], s[4 * h + 1], inv_scale)
                | (mxfp8_e4m3x2(s[4 * h + 2], s[4 * h + 3], inv_scale) << 16);
        }
        *reinterpret_cast<uint2*>(out_q + static_cast<int64_t>(t) * H + c0) = make_uint2(packed[0], packed[1]);
        if ((j & 3) == 0)
        {
            int const kb = c0 / 32;
            out_sf[static_cast<int64_t>(t / 128) * sf_k_tiles * 512 + (kb / 4) * 512 + (t % 32) * 16
                + ((t % 128) / 32) * 4 + kb % 4]
                = static_cast<uint8_t>(code);
        }
    }
#elif defined(__CUDA_ARCH__)
    __trap(); // compiled for an architecture without the sm_10x features: never a silent no-op
#endif
}

void launchRow(
    MinimaxM3Sm107Tp8McAllreduceNormParams const& p, int32_t maxTokens, bool drafter, bool pdl, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(p.hidden == kRowThreads * 8 && p.world == 2, "the row kernel takes H 6144 at 2 ranks");
    static bool const binaryOk
        = checkBinary(mc_allreduce_norm_row_kernel<2, kRowThreads, false, false>, "mc_allreduce_norm_row_kernel")
        && checkBinary(
            mc_allreduce_norm_row_kernel<2, kRowThreads, true, false>, "mc_allreduce_norm_row_kernel (drafter)")
        && checkBinary(
            mc_allreduce_norm_row_kernel<2, kRowThreads, false, true>, "mc_allreduce_norm_row_kernel (capture)");
    (void) binaryOk;
    TLLM_CHECK_WITH_INFO(p.numTokens >= 1 && p.numTokens <= maxTokens, "%d tokens do not fit the %d-token bucket",
        p.numTokens, maxTokens);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.numTokens);
    cfg.blockDim = dim3(kRowThreads);
    cfg.stream = stream;
    if (p.capOut != nullptr)
    {
        TLLM_CHECK_WITH_INFO(
            !drafter && p.hidden == kCapHidden, "the capture epilogue takes H %d, not the drafter", kCapHidden);
        cfg.dynamicSmemBytes = static_cast<size_t>(kRowThreads) * sizeof(uint4);
    }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    // The capture epilogue never runs on the drafter boundary (checked above), so there are three instantiations.
    auto kernel = drafter     ? mc_allreduce_norm_row_kernel<2, kRowThreads, true, false>
        : p.capOut != nullptr ? mc_allreduce_norm_row_kernel<2, kRowThreads, false, true>
                              : mc_allreduce_norm_row_kernel<2, kRowThreads, false, false>;
    cudaError_t const err = cudaLaunchKernelEx(&cfg, kernel, p.x, p.x2, p.hidden, p.residual, p.normWeight, p.eps,
        p.wsUc, p.wsMc, p.flags, p.rank, maxTokens, p.outNormed, p.outResidual, p.outQuant, p.outSf,
        (p.hidden / 32 + 3) / 4, p.prepushed, p.capOut, p.capStride, p.capWeight, p.capEps);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TLLM_CUDA_CHECK(err);
    }
}

constexpr int kCluster = 8;
constexpr int kThreads = 128;

} // namespace

bool isMinimaxM3Sm107Tp8McAllreduceNormSupported(int32_t maxTokens, int32_t opts)
{
    return (maxTokens == 8 && (opts == 7 || opts == 15 || opts == 23 || opts == 31)) || (maxTokens == 16 && opts == 15)
        || (maxTokens == 48 && opts == 15);
}

void invokeMinimaxM3Sm107Tp8McAllreduceNorm(MinimaxM3Sm107Tp8McAllreduceNormParams const& params, int32_t maxTokens,
    int32_t opts, bool pdl, cudaStream_t stream)
{
    if (params.fc2 == nullptr && params.world == 2 && params.hidden == kRowThreads * 8)
    {
        launchRow(params, maxTokens, (opts & 16) != 0, pdl, stream);
        return;
    }
    if (maxTokens == 8 && opts == 7)
    {
        launch<kCluster, kThreads, 8, 7>(params, pdl, stream);
    }
    else if (maxTokens == 8 && opts == 15)
    {
        launch<kCluster, kThreads, 8, 15>(params, pdl, stream);
    }
    else if (maxTokens == 8 && opts == 23)
    {
        launch<kCluster, kThreads, 8, 23>(params, pdl, stream);
    }
    else if (maxTokens == 8 && opts == 31)
    {
        launch<kCluster, kThreads, 8, 31>(params, pdl, stream);
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
