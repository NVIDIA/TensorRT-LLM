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

#include "tensorrt_llm/kernels/minimaxM3/routerGemv.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

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

namespace
{

constexpr int kMaxM = 16;

__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void bulk_g2s(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                     smem_u32(dst)),
                 "l"(src), "r"(bytes), "r"(smem_u32(bar))
                 : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity)
{
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra "
        "WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ float rcp_approx_ftz(float a)
{
    float b;
    asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ uint32_t e2m1x8(float2 const (&v)[4])
{
    uint32_t r;
    asm("{\n .reg .b8 b0, b1, b2, b3;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        " mov.b32 %0, {b0, b1, b2, b3};\n}\n"
        : "=r"(r)
        : "f"(v[0].x), "f"(v[0].y), "f"(v[1].x), "f"(v[1].y), "f"(v[2].x), "f"(v[2].y), "f"(v[3].x), "f"(v[3].y));
    return r;
}

// NVFP4 (16-element E4M3 block scale, per-tensor global scale) quantization of one 16-value block, bit-identical
// to trtllm fp4_quantize (cvt_warp_fp16_to_fp4, linear scale layout).
__device__ __forceinline__ void quant_block16(__nv_bfloat16 const* src, float sf_scale, uint2* q, uint8_t* sf)
{
    uint4 const v0 = *reinterpret_cast<uint4 const*>(src);
    uint4 const v1 = *reinterpret_cast<uint4 const*>(src + 8);
    __nv_bfloat162 h[8];
    *reinterpret_cast<uint4*>(&h[0]) = v0;
    *reinterpret_cast<uint4*>(&h[4]) = v1;
    __nv_bfloat162 mx = __habs2(h[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i)
        mx = __hmax2(mx, __habs2(h[i]));
    float const vec_max = fmaxf(__low2float(mx), __high2float(mx));
    float sf_value = sf_scale * (vec_max * rcp_approx_ftz(6.0f));
    __nv_fp8_e4m3 const tmp = __nv_fp8_e4m3(sf_value);
    sf_value = static_cast<float>(tmp);
    float const out_scale = vec_max != 0.f ? rcp_approx_ftz(sf_value * rcp_approx_ftz(sf_scale)) : 0.f;
    float2 lo[4], hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        lo[i] = __bfloat1622float2(h[i]);
        lo[i].x *= out_scale;
        lo[i].y *= out_scale;
        hi[i] = __bfloat1622float2(h[i + 4]);
        hi[i].x *= out_scale;
        hi[i].y *= out_scale;
    }
    *q = make_uint2(e2m1x8(lo), e2m1x8(hi));
    *sf = tmp.__x;
}

// grid: E / ROWS; block: 256 threads. smem: ROWS * K fp32 weights + M * K bf16 activations.
// With q_out, the CTAs also emit the NVFP4 activation ([M, K / 2] packed, [M, K / 16] E4M3 scales) for the MoE.
template <int ROWS>
__global__ void __launch_bounds__(256) m3_router_gemv_kernel(__nv_bfloat16 const* __restrict__ x, int ldx,
    float const* __restrict__ w, float* __restrict__ out, int M, int K, int E, uint8_t* __restrict__ q_out,
    uint8_t* __restrict__ sf_out, float const* __restrict__ sf_scale)
{
#if MINIMAX_M3_KERNELS_ENABLED
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t w_bar;
    __shared__ __align__(8) uint64_t x_bar;
    __shared__ float red[8][ROWS][kMaxM];
    float* w_s = reinterpret_cast<float*>(smem);
    __nv_bfloat16* x_s = reinterpret_cast<__nv_bfloat16*>(smem + ROWS * K * sizeof(float));
    int const e0 = blockIdx.x * ROWS;
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    if (threadIdx.x == 0)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&w_bar)));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&x_bar)));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&w_bar)),
                     "r"(static_cast<uint32_t>(ROWS * K * sizeof(float)))
                     : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&x_bar)),
                     "r"(static_cast<uint32_t>(M * K * sizeof(__nv_bfloat16)))
                     : "memory");
    }
    __syncthreads();
    // Router weights do not depend on the producer kernel.
    if (warp == 0 && lane < ROWS)
    {
        bulk_g2s(w_s + lane * K, w + static_cast<int64_t>(e0 + lane) * K, K * sizeof(float), &w_bar);
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    if (warp == 1 && lane < M)
    {
        bulk_g2s(x_s + lane * K, x + static_cast<int64_t>(lane) * ldx, K * sizeof(__nv_bfloat16), &x_bar);
    }
    mbar_wait(&x_bar, 0);
    if (q_out != nullptr)
    {
        float const s = *sf_scale;
        int const kblocks = K / 16;
        for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < M * kblocks; b += gridDim.x * blockDim.x)
        {
            int const m = b / kblocks;
            int const kb = b - m * kblocks;
            quant_block16(x_s + m * K + kb * 16, s,
                reinterpret_cast<uint2*>(q_out + static_cast<int64_t>(m) * (K / 2) + kb * 8),
                sf_out + static_cast<int64_t>(m) * kblocks + kb);
        }
    }
    mbar_wait(&w_bar, 0);

    float acc[ROWS][kMaxM];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
    {
#pragma unroll
        for (int m = 0; m < kMaxM; ++m)
        {
            acc[r][m] = 0.f;
        }
    }
    // Each thread owns 4 consecutive columns per 1024-column stride.
    for (int k = threadIdx.x * 4; k < K; k += 256 * 4)
    {
        float4 wv[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            wv[r] = *reinterpret_cast<float4 const*>(w_s + r * K + k);
        }
#pragma unroll
        for (int m = 0; m < kMaxM; ++m)
        {
            if (m < M)
            {
                uint2 const raw = *reinterpret_cast<uint2 const*>(x_s + m * K + k);
                __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&raw);
                float2 const a = __bfloat1622float2(h[0]);
                float2 const b = __bfloat1622float2(h[1]);
#pragma unroll
                for (int r = 0; r < ROWS; ++r)
                {
                    acc[r][m] = fmaf(a.x, wv[r].x, acc[r][m]);
                    acc[r][m] = fmaf(a.y, wv[r].y, acc[r][m]);
                    acc[r][m] = fmaf(b.x, wv[r].z, acc[r][m]);
                    acc[r][m] = fmaf(b.y, wv[r].w, acc[r][m]);
                }
            }
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
    {
#pragma unroll
        for (int m = 0; m < kMaxM; ++m)
        {
            if (m < M)
            {
                float v = acc[r][m];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                {
                    v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
                }
                if (lane == 0)
                {
                    red[warp][r][m] = v;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < ROWS * M)
    {
        int const r = threadIdx.x / M;
        int const m = threadIdx.x % M;
        float s = 0.f;
#pragma unroll
        for (int ww = 0; ww < 8; ++ww)
        {
            s += red[ww][r][m];
        }
        out[static_cast<int64_t>(m) * E + e0 + r] = s;
    }
#endif
}

// ---------------- 9..48 tokens: TF32 tensor-core router (same precision as the native TF32 GEMM) ----------------
// grid: 8 expert tiles (16 experts) x CL CTAs, one CL-CTA cluster per tile, 256 threads. CTA r of a cluster computes
// the [16 experts x M] partials of its 16 / CL K slices (16 slices of 384), each over the same 8-warp split. CTA r
// owns 16 / CL experts of the tile: every CTA st.asyncs its partials for those experts into the owner's mailbox, and
// the owner sums the 16 slices in slice order, so the logits do not depend on CL. Optionally each CTA also emits the
// NVFP4 activation for its K range and the token rows m with m % 8 == expert tile.
constexpr int kTcExperts = 16;
constexpr int kTcSlices = 16;
constexpr int kTcMaxM = 48;

__device__ __forceinline__ uint32_t f2tf32(float f)
{
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(f));
    return r;
}

__device__ __forceinline__ uint32_t mapa_u32(void const* p, uint32_t rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(smem_u32(p)), "r"(rank));
    return r;
}

template <int CL>
__global__ void __launch_bounds__(256) m3_router_tc_kernel(__nv_bfloat16 const* __restrict__ x, int ldx,
    float const* __restrict__ w, float* __restrict__ out, int M, int K, uint8_t* __restrict__ q_out,
    uint8_t* __restrict__ sf_out, float const* __restrict__ sf_scale)
{
#if MINIMAX_M3_KERNELS_ENABLED
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t w_bar;
    __shared__ __align__(8) uint64_t x_bar;
    __shared__ __align__(8) uint64_t mb_bar;
    constexpr int SPC = kTcSlices / CL;              // K slices per CTA
    constexpr int EPC = kTcExperts / CL;             // experts owned per CTA
    __shared__ float red[8][kTcExperts][kTcMaxM];
    __shared__ float part[SPC][kTcExperts][kTcMaxM]; // this CTA's slice partials
    __shared__ float mailbox[CL][SPC][EPC][kTcMaxM]; // [source CTA][slice][owned expert][m]
    int const tile = blockIdx.x / CL;
    int const crank = blockIdx.x % CL;
    int const ks = K / kTcSlices; // 384
    int const kc = SPC * ks;
    int const k0 = crank * kc;
    int const e0 = tile * kTcExperts;
    int const wst = kc + 4;                      // padded row strides (bank-conflict-free fragment loads)
    int const xst = kc + 8;
    float* w_s = reinterpret_cast<float*>(smem); // [16][wst]
    __nv_bfloat16* x_s = reinterpret_cast<__nv_bfloat16*>(smem + kTcExperts * wst * 4); // [M][xst]
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    if (threadIdx.x == 0)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&w_bar)));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&x_bar)));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&mb_bar)));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&w_bar)),
                     "r"(static_cast<uint32_t>(kTcExperts * kc * 4))
                     : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&x_bar)),
                     "r"(static_cast<uint32_t>(M * kc * 2))
                     : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&mb_bar)),
                     "r"(static_cast<uint32_t>((CL - 1) * SPC * EPC * M * 4))
                     : "memory");
    }
    __syncthreads();
    // The peers' mailbox barriers must be initialized before this CTA's st.async lands there.
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    if (warp == 0 && lane < kTcExperts)
        bulk_g2s(w_s + lane * wst, w + static_cast<int64_t>(e0 + lane) * K + k0, kc * 4, &w_bar);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    if (warp == 1)
    {
        for (int m = lane; m < M; m += 32)
            bulk_g2s(x_s + m * xst, x + static_cast<int64_t>(m) * ldx + k0, kc * 2, &x_bar);
    }
    mbar_wait(&x_bar, 0);
    if (q_out != nullptr)
    {
        float const sc = *sf_scale;
        int const kb_cta = kc / 16;
        for (int b = threadIdx.x; b < M * kb_cta; b += blockDim.x)
        {
            int const m = b / kb_cta;
            if (m % 8 != tile)
                continue; // 8 expert tiles share the row quantization
            int const kb = b - m * kb_cta;
            int const kbg = k0 / 16 + kb;
            quant_block16(x_s + m * xst + kb * 16, sc,
                reinterpret_cast<uint2*>(q_out + static_cast<int64_t>(m) * (K / 2) + kbg * 8),
                sf_out + static_cast<int64_t>(m) * (K / 16) + kbg);
        }
    }
    mbar_wait(&w_bar, 0);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    // Per slice h: warp w owns k8 steps [w * steps, (w + 1) * steps) of the slice; all token tiles (n = 8 each).
    int const g = lane >> 2;
    int const t = lane & 3;
    int const ntiles = (M + 7) / 8;
    int const steps = ks / 8 / 8;
#pragma unroll 1
    for (int h = 0; h < SPC; ++h)
    {
        float acc[6][4];
#pragma unroll
        for (int i = 0; i < 6; ++i)
            acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
        for (int st = warp * steps; st < (warp + 1) * steps; ++st)
        {
            int const kk = h * ks + st * 8;
            uint32_t const a0 = f2tf32(w_s[g * wst + kk + t]);
            uint32_t const a1 = f2tf32(w_s[(g + 8) * wst + kk + t]);
            uint32_t const a2 = f2tf32(w_s[g * wst + kk + t + 4]);
            uint32_t const a3 = f2tf32(w_s[(g + 8) * wst + kk + t + 4]);
#pragma unroll
            for (int nt = 0; nt < 6; ++nt)
            {
                if (nt < ntiles)
                {
                    int const m = nt * 8 + g;
                    float bx0 = 0.f, bx1 = 0.f;
                    if (m < M)
                    {
                        bx0 = __bfloat162float(x_s[m * xst + kk + t]);
                        bx1 = __bfloat162float(x_s[m * xst + kk + t + 4]);
                    }
                    uint32_t const b0 = f2tf32(bx0), b1 = f2tf32(bx1);
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                        "{%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
        {
            if (nt < ntiles)
            {
                int const c = nt * 8 + 2 * t;
                red[warp][g][c] = acc[nt][0];
                red[warp][g][c + 1] = acc[nt][1];
                red[warp][g + 8][c] = acc[nt][2];
                red[warp][g + 8][c + 1] = acc[nt][3];
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < kTcExperts * M; i += blockDim.x)
        {
            int const e = i / M, m = i % M;
            float v = 0.f;
#pragma unroll
            for (int ww = 0; ww < 8; ++ww)
                v += red[ww][e][m];
            part[h][e][m] = v;
        }
        __syncthreads();
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    // Expert e of the tile belongs to CTA e / EPC: its slice partials go to that CTA's mailbox[crank].
    for (int i = threadIdx.x; i < SPC * kTcExperts * M; i += blockDim.x)
    {
        int const h = i / (kTcExperts * M);
        int const r = i - h * kTcExperts * M;
        int const e = r / M, m = r % M;
        int const owner = e / EPC;
        float const v = part[h][e][m];
        float* dst = &mailbox[crank][h][e % EPC][m];
        if (owner == crank)
        {
            *dst = v;
        }
        else
        {
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(
                             mapa_u32(dst, owner)),
                         "r"(__float_as_uint(v)), "r"(mapa_u32(&mb_bar, owner))
                         : "memory");
        }
    }
    __syncthreads();
    {
        uint32_t done = 0;
        while (!done)
        {
            asm volatile(
                "{\n .reg .pred p;\n mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
                " selp.u32 %0, 1, 0, p;\n}\n"
                : "=r"(done)
                : "r"(smem_u32(&mb_bar))
                : "memory");
        }
    }
    for (int i = threadIdx.x; i < EPC * M; i += blockDim.x)
    {
        int const le = i / M, m = i % M;
        float v = 0.f;
#pragma unroll
        for (int src = 0; src < CL; ++src)
        {
#pragma unroll
            for (int h = 0; h < SPC; ++h)
                v += mailbox[src][h][le][m];
        }
        out[static_cast<int64_t>(m) * (kTcExperts * 8) + e0 + EPC * crank + le] = v;
    }
#endif
}

template <int ROWS>
void launchGemv(__nv_bfloat16 const* x, int ldx, int M, int K, float const* w, int E, float* out, uint8_t* q_out,
    uint8_t* sf_out, float const* sf_scale, bool pdl, cudaStream_t stream)
{
    size_t const smem = static_cast<size_t>(ROWS) * K * sizeof(float) + static_cast<size_t>(M) * K * 2;
    auto kernel = m3_router_gemv_kernel<ROWS>;
    static bool const configured = [&]()
    {
        int device = 0, max_optin = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        cudaFuncAttributes attrs = {};
        if (cudaFuncGetAttributes(&attrs, kernel) != cudaSuccess)
        {
            return false;
        }
        return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   max_optin - static_cast<int>(attrs.sharedSizeBytes))
            == cudaSuccess;
    }();
    TLLM_CHECK_WITH_INFO(configured, "failed to configure dynamic shared memory");
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(E / ROWS);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t const err = cudaLaunchKernelEx(&cfg, kernel, x, ldx, w, out, M, K, E, q_out, sf_out, sf_scale);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TLLM_CUDA_CHECK(err);
    }
}

} // namespace

void invokeMinimaxM3RouterGemv(__nv_bfloat16 const* x, int32_t ldx, int32_t m, int32_t k, float const* w, int32_t e,
    float* out, uint8_t* qOut, uint8_t* sfOut, float const* sfScale, bool pdl, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(m >= 1 && m <= kMaxM && k % 1024 == 0, "unsupported router GEMV shape M=%d K=%d", m, k);
    launchGemv<1>(x, ldx, m, k, w, e, out, qOut, sfOut, sfScale, pdl, stream);
}

void invokeMinimaxM3RouterTc(__nv_bfloat16 const* x, int32_t ldx, int32_t m, int32_t k, float const* w, float* out,
    uint8_t* qOut, uint8_t* sfOut, float const* sfScale, bool pdl, cudaStream_t stream)
{
    int64_t const M = m, K = k;
    TLLM_CHECK_WITH_INFO(
        M >= 1 && M <= kTcMaxM && K % (kTcSlices * 64) == 0, "unsupported router shape M=%d K=%d", m, k);
    int const ks = K / kTcSlices;
    auto launch_tc = [&](auto kernel, int cl) -> cudaError_t
    {
        int const kc = (kTcSlices / cl) * ks;
        size_t const smem = static_cast<size_t>(kTcExperts) * (kc + 4) * 4 + static_cast<size_t>(M) * (kc + 8) * 2;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(8 * cl);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[2];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
        attr[1].id = cudaLaunchAttributeClusterDimension;
        attr[1].val.clusterDim.x = cl;
        attr[1].val.clusterDim.y = 1;
        attr[1].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 2;
        return cudaLaunchKernelEx(
            &cfg, kernel, x, ldx, w, out, static_cast<int>(M), static_cast<int>(K), qOut, sfOut, sfScale);
    };
    // 16-CTA clusters (128 CTAs, one K slice each) when the device co-schedules all 8, else 8-CTA clusters.
    static int const cluster = [&]()
    {
        int device = 0, max_optin = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        auto configure = [&](auto kernel, bool non_portable)
        {
            cudaFuncAttributes attrs = {};
            if (cudaFuncGetAttributes(&attrs, kernel) != cudaSuccess)
                return false;
            if (non_portable
                && cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess)
                return false;
            return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_optin - static_cast<int>(attrs.sharedSizeBytes))
                == cudaSuccess;
        };
        TLLM_CHECK_WITH_INFO(configure(m3_router_tc_kernel<8>, false), "failed to configure the router kernel");
        if (!configure(m3_router_tc_kernel<16>, true))
        {
            cudaGetLastError();
            return 8;
        }
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(8 * 16);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes
            = static_cast<size_t>(kTcExperts) * (ks + 4) * 4 + static_cast<size_t>(kTcMaxM) * (ks + 8) * 2;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 16;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        int clusters = 0;
        if (cudaOccupancyMaxActiveClusters(&clusters, m3_router_tc_kernel<16>, &cfg) != cudaSuccess)
        {
            cudaGetLastError();
            return 8;
        }
        return clusters >= 8 ? 16 : 8;
    }();
    cudaError_t const err
        = cluster == 16 ? launch_tc(m3_router_tc_kernel<16>, 16) : launch_tc(m3_router_tc_kernel<8>, 8);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TLLM_CUDA_CHECK(err);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
