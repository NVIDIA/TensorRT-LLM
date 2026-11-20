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

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/nvfp4W4a16Gemv.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Cuda-core W4A16 NVFP4 GEMV for decode rows (see nvfp4W4a16Gemv.h).
//
// A CTA owns kGemvRows output rows across the whole K; thread t owns the 32
// K-elements [32 t, 32 t + 32) (K == 32 * BLOCK), i.e. 16 weight bytes and two
// E4M3 block scales per row. Every weight/scale load is issued before
// griddepcontrol.wait (they do not depend on the producer of x), and
// dependents are triggered once the FMAs are done, before the reduction. The
// summation order is fixed (per 16-block FP32 partials, warp xor-shuffle tree,
// warps in index order), so repeated calls are bitwise identical.

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kGemvRows = 2;

template <int M, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
    nvfp4W4a16GemvKernel(uint8_t const* __restrict__ w, uint8_t const* __restrict__ sf,
        __nv_bfloat16 const* __restrict__ x, __nv_bfloat16* __restrict__ out, float g, int n, int k)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    int const tid = threadIdx.x;
    int const n0 = blockIdx.x * kGemvRows;
    int const kb = k / 2;   // weight row bytes
    int const ksf = k / 16; // scale row bytes

    int4 wq[kGemvRows];
    uint32_t ws[kGemvRows];
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        uint8_t const* p = w + static_cast<size_t>(n0 + r) * kb + tid * 16;
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(wq[r].x), "=r"(wq[r].y), "=r"(wq[r].z), "=r"(wq[r].w)
                     : "l"(p));
    }
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        ws[r] = __ldg(reinterpret_cast<uint16_t const*>(sf + static_cast<size_t>(n0 + r) * ksf + tid * 2));
    }

    // x is produced by the previous kernel.
    asm volatile("griddepcontrol.wait;" ::: "memory");

    uint32_t xa[M][16];
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
        int4 const* p = reinterpret_cast<int4 const*>(x + static_cast<size_t>(m) * k + tid * 32);
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            int4 const v = p[q];
            xa[m][q * 4 + 0] = v.x;
            xa[m][q * 4 + 1] = v.y;
            xa[m][q * 4 + 2] = v.z;
            xa[m][q * 4 + 3] = v.w;
        }
    }

    float acc[M][kGemvRows];
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int r = 0; r < kGemvRows; ++r)
        {
            acc[m][r] = 0.f;
        }
    }

#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        // 16 packed E2M1 bytes -> 32 floats (element 2b = low nibble of byte b).
        float wf[32];
        uint32_t const* wv = reinterpret_cast<uint32_t const*>(&wq[r]);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            uint32_t h[4];
            asm volatile(
                "{\n.reg .b8 b0, b1, b2, b3;\n"
                "mov.b32 {b0, b1, b2, b3}, %4;\n"
                "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
                "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
                "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
                "cvt.rn.f16x2.e2m1x2 %3, b3;\n}\n"
                : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
                : "r"(wv[i]));
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float2 const f = __half22float2(reinterpret_cast<__half2 const&>(h[j]));
                wf[i * 8 + j * 2] = f.x;
                wf[i * 8 + j * 2 + 1] = f.y;
            }
        }
        __nv_fp8_e4m3 s8;
        s8.__x = static_cast<__nv_fp8_storage_t>(ws[r] & 0xff);
        float const s0 = static_cast<float>(s8);
        s8.__x = static_cast<__nv_fp8_storage_t>((ws[r] >> 8) & 0xff);
        float const s1 = static_cast<float>(s8);
#pragma unroll
        for (int m = 0; m < M; ++m)
        {
            float p0 = 0.f;
            float p1 = 0.f;
#pragma unroll
            for (int l = 0; l < 8; ++l)
            {
                p0 = fmaf(__uint_as_float(xa[m][l] << 16), wf[2 * l], p0);
                p0 = fmaf(__uint_as_float(xa[m][l] & 0xffff0000u), wf[2 * l + 1], p0);
            }
#pragma unroll
            for (int l = 8; l < 16; ++l)
            {
                p1 = fmaf(__uint_as_float(xa[m][l] << 16), wf[2 * l], p1);
                p1 = fmaf(__uint_as_float(xa[m][l] & 0xffff0000u), wf[2 * l + 1], p1);
            }
            acc[m][r] = fmaf(p0, s0, acc[m][r]);
            acc[m][r] = fmaf(p1, s1, acc[m][r]);
        }
    }

    // Dependents still wait for this grid's completion before reading out.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    constexpr int kWarps = BLOCK / 32;
    __shared__ float red[kWarps][M * kGemvRows];
    int const lane = tid & 31;
    int const warp = tid >> 5;
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int r = 0; r < kGemvRows; ++r)
        {
            float v = acc[m][r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                v += __shfl_xor_sync(0xffffffffu, v, o);
            }
            if (lane == 0)
            {
                red[warp][m * kGemvRows + r] = v;
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < M * kGemvRows; i += BLOCK)
    {
        float v = 0.f;
#pragma unroll
        for (int j = 0; j < kWarps; ++j)
        {
            v += red[j][i];
        }
        int const m = i / kGemvRows;
        int const r = i % kGemvRows;
        out[static_cast<size_t>(m) * n + n0 + r] = __float2bfloat16(v * g);
    }
#endif
}

// GEMV + fused GDN decode causal_conv1d_update epilogue (see nvfp4_w4a16_gemv.h). The weight/SF loads,
// dependency wait, FMA loop, early trigger and reduction are those of nvfp4W4a16GemvKernel; thread
// tid < M * kGemvRows then owns output (row tid / 2, column n0 + tid % 2) and, for conv columns, applies
// causal_conv1d_update_kernel_sl1's math with the same operation order and intrinsics:
//   out = 0; out = fma(w[i], s[i], out) for i = 0..3 (s = [state0, state1, state2, x]);
//   out = out * rcp_rn(1 + __expf(-out)); state = [state1, state2, x].
// Only the constant conv weight is read before griddepcontrol.wait; the slot index and conv state are
// read after it. Two load schedules, chosen by M:
//  - EARLY (M == 1): weight (pre-wait) and state (post-wait, beside the x loads) are held as packed BF16
//    across the FMAs, so the tail only adds the last tap.
//  - LATE (M >= 2, where those registers cost occupancy): the conv lanes load weight and state after
//    the FMAs, fold the three state taps (same fma order) and store the shifted state[0..1] before the
//    reduction; the tail adds the x tap and stores state[2].
template <int M, int BLOCK, bool EARLY, int MINB>
__global__ void __launch_bounds__(BLOCK, MINB) nvfp4W4a16GemvConvKernel(uint8_t const* __restrict__ w,
    uint8_t const* __restrict__ sf, __nv_bfloat16 const* __restrict__ x, __nv_bfloat16* __restrict__ out, float g,
    int n, int k, __nv_bfloat16* __restrict__ convState, __nv_bfloat16 const* __restrict__ convWeight,
    int const* __restrict__ convIndices, int64_t convStateBatchStride, int convDim, int padSlotId)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(M * kGemvRows <= BLOCK, "one epilogue element per thread");
    int const tid = threadIdx.x;
    int const n0 = blockIdx.x * kGemvRows;
    int const kb = k / 2;   // weight row bytes
    int const ksf = k / 16; // scale row bytes

    int4 wq[kGemvRows];
    uint32_t ws[kGemvRows];
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        uint8_t const* p = w + static_cast<size_t>(n0 + r) * kb + tid * 16;
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(wq[r].x), "=r"(wq[r].y), "=r"(wq[r].z), "=r"(wq[r].w)
                     : "l"(p));
    }
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        ws[r] = __ldg(reinterpret_cast<uint16_t const*>(sf + static_cast<size_t>(n0 + r) * ksf + tid * 2));
    }

    // convDim is even, so a CTA's two columns are both conv channels or both z channels.
    bool const convThread = n0 < convDim && tid < M * kGemvRows;
    int const ch = n0 + (tid & 1);
    // Conv weight [convDim, 4] BF16, contiguous: 8 bytes per channel.
    uint2 cw = make_uint2(0u, 0u);
    if constexpr (EARLY)
    {
        if (convThread)
        {
            // Constant across the step: safe to read before the dependency wait.
            cw = __ldg(reinterpret_cast<uint2 const*>(convWeight) + ch);
        }
    }
    else
    {
        if (n0 < convDim && tid == 0)
        {
            // The CTA's two conv weight rows (16 contiguous bytes); constant, so safe before the wait.
            asm volatile("prefetch.global.L1 [%0];" ::"l"(reinterpret_cast<uint2 const*>(convWeight) + n0));
        }
    }

    // x (and the conv state / slot indices) are produced or updated by earlier kernels.
    asm volatile("griddepcontrol.wait;" ::: "memory");

    uint32_t xa[M][16];
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
        int4 const* p = reinterpret_cast<int4 const*>(x + static_cast<size_t>(m) * k + tid * 32);
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            int4 const v = p[q];
            xa[m][q * 4 + 0] = v.x;
            xa[m][q * 4 + 1] = v.y;
            xa[m][q * 4 + 2] = v.z;
            xa[m][q * 4 + 3] = v.w;
        }
    }

    // Conv state [slots, convDim, 3] BF16 with strides (convStateBatchStride, 3, 1).
    int slot = padSlotId;
    unsigned short* sp = nullptr;
    if (convThread)
    {
        slot = convIndices[tid / kGemvRows];
        if (slot != padSlotId)
        {
            sp = reinterpret_cast<unsigned short*>(convState) + static_cast<int64_t>(slot) * convStateBatchStride
                + static_cast<int64_t>(ch) * 3;
        }
    }
    uint32_t s01 = 0u;
    uint32_t s2 = 0u;
    if constexpr (EARLY)
    {
        if (sp != nullptr)
        {
            s01 = static_cast<uint32_t>(sp[0]) | (static_cast<uint32_t>(sp[1]) << 16);
            s2 = static_cast<uint32_t>(sp[2]);
        }
    }
    else
    {
        if (sp != nullptr)
        {
            asm volatile("prefetch.global.L1 [%0];" ::"l"(sp));
        }
    }

    float acc[M][kGemvRows];
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int r = 0; r < kGemvRows; ++r)
        {
            acc[m][r] = 0.f;
        }
    }

#pragma unroll
    for (int r = 0; r < kGemvRows; ++r)
    {
        // 16 packed E2M1 bytes -> 32 floats (element 2b = low nibble of byte b).
        float wf[32];
        uint32_t const* wv = reinterpret_cast<uint32_t const*>(&wq[r]);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            uint32_t h[4];
            asm volatile(
                "{\n.reg .b8 b0, b1, b2, b3;\n"
                "mov.b32 {b0, b1, b2, b3}, %4;\n"
                "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
                "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
                "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
                "cvt.rn.f16x2.e2m1x2 %3, b3;\n}\n"
                : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
                : "r"(wv[i]));
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float2 const f = __half22float2(reinterpret_cast<__half2 const&>(h[j]));
                wf[i * 8 + j * 2] = f.x;
                wf[i * 8 + j * 2 + 1] = f.y;
            }
        }
        __nv_fp8_e4m3 s8;
        s8.__x = static_cast<__nv_fp8_storage_t>(ws[r] & 0xff);
        float const s0 = static_cast<float>(s8);
        s8.__x = static_cast<__nv_fp8_storage_t>((ws[r] >> 8) & 0xff);
        float const s1 = static_cast<float>(s8);
#pragma unroll
        for (int m = 0; m < M; ++m)
        {
            float p0 = 0.f;
            float p1 = 0.f;
#pragma unroll
            for (int l = 0; l < 8; ++l)
            {
                p0 = fmaf(__uint_as_float(xa[m][l] << 16), wf[2 * l], p0);
                p0 = fmaf(__uint_as_float(xa[m][l] & 0xffff0000u), wf[2 * l + 1], p0);
            }
#pragma unroll
            for (int l = 8; l < 16; ++l)
            {
                p1 = fmaf(__uint_as_float(xa[m][l] << 16), wf[2 * l], p1);
                p1 = fmaf(__uint_as_float(xa[m][l] & 0xffff0000u), wf[2 * l + 1], p1);
            }
            acc[m][r] = fmaf(p0, s0, acc[m][r]);
            acc[m][r] = fmaf(p1, s1, acc[m][r]);
        }
    }

    // Dependents still wait for this grid's completion before reading out or the conv state.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    if constexpr (!EARLY)
    {
        // LATE: weight and state are read here, after the FMAs (and after the wait).
        if (sp != nullptr)
        {
            cw = __ldg(reinterpret_cast<uint2 const*>(convWeight) + ch);
            s01 = static_cast<uint32_t>(sp[0]) | (static_cast<uint32_t>(sp[1]) << 16);
            s2 = static_cast<uint32_t>(sp[2]);
        }
    }
    // Conv taps 0..2 (the state), in causal_conv1d_update_kernel_sl1's order; tap 3 (x) follows the
    // reduction. LATE folds them here, before the reduction; EARLY in the tail.
    float part = 0.f;
    auto foldState = [&]()
    {
        part = __fmaf_rn(__uint_as_float(cw.x << 16), __uint_as_float(s01 << 16), part);
        part = __fmaf_rn(__uint_as_float(cw.x & 0xffff0000u), __uint_as_float(s01 & 0xffff0000u), part);
        part = __fmaf_rn(__uint_as_float(cw.y << 16), __uint_as_float(s2 << 16), part);
        // state = [state1, state2, x]: words 0/1 here, word 2 (x) in the tail.
        sp[0] = __bfloat16_as_ushort(__float2bfloat16(__uint_as_float(s01 & 0xffff0000u)));
        sp[1] = __bfloat16_as_ushort(__float2bfloat16(__uint_as_float(s2 << 16)));
    };
    if constexpr (!EARLY)
    {
        if (sp != nullptr)
        {
            foldState();
        }
    }

    constexpr int kWarps = BLOCK / 32;
    __shared__ float red[kWarps][M * kGemvRows];
    int const lane = tid & 31;
    int const warp = tid >> 5;
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int r = 0; r < kGemvRows; ++r)
        {
            float v = acc[m][r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                v += __shfl_xor_sync(0xffffffffu, v, o);
            }
            if (lane == 0)
            {
                red[warp][m * kGemvRows + r] = v;
            }
        }
    }
    __syncthreads();
    if (tid < M * kGemvRows)
    {
        float v = 0.f;
#pragma unroll
        for (int j = 0; j < kWarps; ++j)
        {
            v += red[j][tid];
        }
        int const m = tid / kGemvRows;
        int const r = tid % kGemvRows;
        __nv_bfloat16 o = __float2bfloat16(v * g);
        if (sp != nullptr)
        {
            if constexpr (EARLY)
            {
                foldState();
            }
            float ov = __fmaf_rn(__uint_as_float(cw.y & 0xffff0000u), __bfloat162float(o), part);
            ov = ov * __frcp_rn(1.0f + __expf(-ov));
            sp[2] = __bfloat16_as_ushort(__float2bfloat16(__bfloat162float(o)));
            o = __float2bfloat16(ov);
        }
        out[static_cast<size_t>(m) * n + n0 + r] = o;
    }
#endif
}

template <int BLOCK>
bool launchNvfp4W4a16Gemv(void const* x, void const* w, void const* sf, void* out, float g, int m, int n, int k,
    bool enablePdl, cudaStream_t stream)
{
    auto const* wp = static_cast<uint8_t const*>(w);
    auto const* sp = static_cast<uint8_t const*>(sf);
    auto const* xp = static_cast<__nv_bfloat16 const*>(x);
    auto* op = static_cast<__nv_bfloat16*>(out);
    dim3 const grid(n / kGemvRows);
    dim3 const block(BLOCK);
    auto launch = [&](auto kernel)
    {
        tensorrt_llm::common::launchWithPdl(
            "nvfp4_w4a16_gemv", enablePdl, kernel, grid, block, 0, stream, wp, sp, xp, op, g, n, k);
        return true;
    };
    switch (m)
    {
    case 1: return launch(&nvfp4W4a16GemvKernel<1, BLOCK>);
    case 2: return launch(&nvfp4W4a16GemvKernel<2, BLOCK>);
    case 3: return launch(&nvfp4W4a16GemvKernel<3, BLOCK>);
    case 4: return launch(&nvfp4W4a16GemvKernel<4, BLOCK>);
    default: return false;
    }
}

template <int BLOCK>
bool launchNvfp4W4a16GemvConv(void const* x, void const* w, void const* sf, void* out, float g, int m, int n, int k,
    Nvfp4GemvConvParams const& conv, bool enablePdl, cudaStream_t stream)
{
    auto const* wp = static_cast<uint8_t const*>(w);
    auto const* sp = static_cast<uint8_t const*>(sf);
    auto const* xp = static_cast<__nv_bfloat16 const*>(x);
    auto* op = static_cast<__nv_bfloat16*>(out);
    auto* cs = static_cast<__nv_bfloat16*>(conv.convState);
    auto const* cw = static_cast<__nv_bfloat16 const*>(conv.convWeight);
    dim3 const grid(n / kGemvRows);
    dim3 const block(BLOCK);
    auto launch = [&](auto kernel)
    {
        tensorrt_llm::common::launchWithPdl("nvfp4_w4a16_gemv_conv", enablePdl, kernel, grid, block, 0, stream, wp, sp,
            xp, op, g, n, k, cs, cw, conv.convStateIndices, conv.convStateBatchStride, conv.convDim, conv.padSlotId);
        return true;
    };
    // Load schedule per M: M=1 holds the conv operands in registers across the FMAs; M=2/3 load them after
    // the FMAs, M=3 with __launch_bounds__(BLOCK, 4).
    switch (m)
    {
    case 1: return launch(&nvfp4W4a16GemvConvKernel<1, BLOCK, true, 1>);
    case 2: return launch(&nvfp4W4a16GemvConvKernel<2, BLOCK, false, 1>);
    case 3: return launch(&nvfp4W4a16GemvConvKernel<3, BLOCK, false, 4>);
    default: return false;
    }
}

} // namespace

bool nvfp4W4a16GemvSupported(int n, int k)
{
    int const sm = tensorrt_llm::common::getSMVersion();
    if (sm != 120 && sm != 121)
    {
        return false;
    }
    return n > 0 && n % kGemvRows == 0 && (k == 6144 || k == 5120);
}

bool nvfp4W4a16Gemv(void const* x, void const* w, void const* sf, void* out, float g, int m, int n, int k,
    bool enablePdl, cudaStream_t stream)
{
    if (m < 1 || m > kNvfp4W4a16GemvMaxRows || !nvfp4W4a16GemvSupported(n, k))
    {
        return false;
    }
    if (k == 6144)
    {
        return launchNvfp4W4a16Gemv<192>(x, w, sf, out, g, m, n, k, enablePdl, stream);
    }
    return launchNvfp4W4a16Gemv<160>(x, w, sf, out, g, m, n, k, enablePdl, stream);
}

bool nvfp4W4a16GemvConv(void const* x, void const* w, void const* sf, void* out, float g, int m, int n, int k,
    Nvfp4GemvConvParams const& conv, bool enablePdl, cudaStream_t stream)
{
    if (m < 1 || m > kNvfp4W4a16GemvConvMaxRows || !nvfp4W4a16GemvSupported(n, k))
    {
        return false;
    }
    if (conv.convDim <= 0 || conv.convDim % kGemvRows != 0 || conv.convDim > n
        || reinterpret_cast<uintptr_t>(conv.convWeight) % 8 != 0)
    {
        return false;
    }
    if (k == 6144)
    {
        return launchNvfp4W4a16GemvConv<192>(x, w, sf, out, g, m, n, k, conv, enablePdl, stream);
    }
    return launchNvfp4W4a16GemvConv<160>(x, w, sf, out, g, m, n, k, conv, enablePdl, stream);
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
