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

#include "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified/csrc/hcAllreduceInjectNorm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/lamportComm.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace hc_inject
{
// This kernel runs on the workspace of the stock one-shot all-reduce `allreduce_fusion_kernel_oneshot_lamport`
// (communicationKernels/allReduceFusionKernels.cu) between its calls, so it takes that file's LamportComm and helpers
// (communicationKernels/lamportComm.cuh) and its push / clear / poll / update steps match that kernel's:
//   - workspace[2 * NRanks + r] is rank r's buffer of three sub-buffers of workspace[3 * NRanks + 1][1] bytes; with
//     flag = workspace[3 * NRanks][2], a call writes and polls sub-buffer flag % 3 and clears its own (flag + 2) % 3.
//   - workspace[3 * NRanks + 1][0] holds the bytes the previous call wrote, cleared as 16-byte stores of -0.0f.
//   - A 16-byte slot has arrived once none of its 32-bit words is -0.0f, so senders store -0.0f as +0.0f.
//   - Every CTA increments workspace[3 * NRanks][0] after reading the flag; once all have, CTA 0 advances the flag to
//     (flag + 1) % 3, stores the bytes this call wrote and resets the counter.
using ::tensorrt_llm::kernels::ar_fusion::get_neg_zero;
using ::tensorrt_llm::kernels::ar_fusion::get_sm_count;
using ::tensorrt_llm::kernels::ar_fusion::is_neg_zero;
using ::tensorrt_llm::kernels::ar_fusion::LamportComm;
using ::tensorrt_llm::kernels::ar_fusion::ld_global_volatile;
} // namespace hc_inject

// Thread t holds columns [8t, 8t + 8) of a row's block output and of each of its 4 streams. The Triton program runs a
// stream on G = 32 * program_warps threads over a 4096-wide block, its thread u in chunk c holding this CTA's thread
// c * G + u. Its sum of squares, reproduced bitwise: per thread and chunk the fp32 pair (fma(x0, x0, x2 * x2) +
// fma(x4, x4, x6 * x6), fma(x1, x1, x3 * x3) + fma(x5, x5, x7 * x7)), a pairwise tree over chunks (masked ones are
// zero), low + high, the warp's xor butterfly 16..1, then ((w0 + w2) + (w1 + w3)) at 4 warps or (w0 + w1) at 2.
namespace hc_inject
{
static constexpr int kHidden = 2560;
static constexpr int kStreams = 4;
static constexpr int kThreads = kHidden / 8;

__device__ __forceinline__ float bf16_lo(uint32_t w)
{
    return __uint_as_float(w << 16);
}

__device__ __forceinline__ float bf16_hi(uint32_t w)
{
    return __uint_as_float(w & 0xFFFF0000u);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi)
{
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ float round_bf16(float x)
{
    return bf16_lo(pack_bf16x2(x, 0.f));
}

__device__ __forceinline__ float ld_bf16(void const* base, int64_t elem)
{
    uint16_t v = reinterpret_cast<uint16_t const*>(base)[elem];
    return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

//! `tl.sigmoid` as the Triton kernel compiles it.
__device__ __forceinline__ float sigmoid_triton(float x)
{
    float y;
    asm("{ .reg .f32 t; mul.f32 t, %1, 0fBFB8AA3B; ex2.approx.f32 t, t; add.f32 t, t, 0f3F800000; "
        "div.full.f32 %0, 0f3F800000, t; }"
        : "=f"(y)
        : "f"(x));
    return y;
}

//! `rsqrt(ss / 2560 + eps)` as the Triton kernel compiles it (`div.full`, `add`, `rsqrt.approx.ftz`).
__device__ __forceinline__ float rstd_triton(float ss, float eps)
{
    float y;
    asm("{ .reg .f32 t; div.full.f32 t, %1, 0f45200000; add.f32 t, t, %2; rsqrt.approx.ftz.f32 %0, t; }"
        : "=f"(y)
        : "f"(ss), "f"(eps));
    return y;
}

__device__ __forceinline__ float2 add2(float2 a, float2 b)
{
    return make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
}

//! A program thread's sum of squares: the pairwise tree over its unmasked chunk pairs, then low + high.
template <int G>
__device__ __forceinline__ float thread_ssq(float2 const* leaves, int u)
{
    if constexpr (G == 128)
    {
        // 4 chunks of 1024 columns: chunk 2 real for u < 64, chunk 3 masked.
        float2 l = add2(leaves[u], leaves[G + u]);
        if (u < 64)
        {
            l = add2(l, leaves[2 * G + u]);
        }
        return __fadd_rn(l.x, l.y);
    }
    else
    {
        static_assert(G == 64, "4- or 2-warp programs");
        // 8 chunks of 512 columns: chunks 0..4 real, 5..7 masked.
        float2 a = add2(leaves[u], leaves[G + u]);
        float2 b = add2(leaves[2 * G + u], leaves[3 * G + u]);
        float2 l = add2(add2(a, b), leaves[4 * G + u]);
        return __fadd_rn(l.x, l.y);
    }
}

//! Launch forms with the same arithmetic: Var 2 asks for three CTAs an SM (one wave of one-row CTAs up to 3 x SMs),
//! Var 3 has no occupancy bound. launch() takes 3 while the rows fit its resident grid, else 2 if they fit 2's, else 3.
template <int Var>
struct HcVariant
{
    static constexpr int kMinBlocks = Var == 2 ? 3 : 1;
};

template <int NRanks, int G, bool HasShared, bool TriggerCompletionAtEnd, int Var>
__global__ void __launch_bounds__(kThreads, HcVariant<Var>::kMinBlocks)
    hc_allreduce_inject_norm_kernel(HcInjectNormParams params)
{
    static_assert(kHidden % (8 * G) == 0 || G == 128, "the chunk layout above");
    __shared__ float2 s_leaf[kStreams][kThreads];
    __shared__ float s_red[kStreams][G / 32];
    __shared__ uint4 s_w[kStreams][kThreads];
    int const tid = threadIdx.x;
    int const lane = tid & 31;

    // No kernel of the step writes the norm weights: stage them before the PDL wait (each thread reads back its own).
#pragma unroll
    for (int s = 0; s < kStreams; ++s)
    {
        s_w[s][tid] = reinterpret_cast<uint4 const*>(params.norm_weight)[s * kThreads + tid];
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    if constexpr (!TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
    LamportComm<NRanks> comm(params.workspace, params.rank);
    int const rows = params.rows;
    int const tot_access = rows * kThreads;
    int const clear_access = static_cast<int>((comm.clear_size + sizeof(float4) - 1) / sizeof(float4));

    // Push this rank's partial rows, -0.0f stored as +0.0f.
    for (int row = blockIdx.x; row < rows; row += gridDim.x)
    {
        int const idx = row * kThreads + tid;
        alignas(16) float val[4];
        *reinterpret_cast<float4*>(val) = reinterpret_cast<float4 const*>(params.partial)[idx];

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (is_neg_zero(val[i]))
            {
                val[i] = 0.f;
            }
        }
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * tot_access + idx]
                = *reinterpret_cast<float4*>(val);
        }
    }
    float4 const clear_vec = get_neg_zero();
    for (int idx = blockIdx.x * kThreads + tid; idx < clear_access; idx += gridDim.x * kThreads)
    {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    for (int row = blockIdx.x; row < rows; row += gridDim.x)
    {
        int const idx = row * kThreads + tid;
        // The site's operands (issued before the poll, consumed after it).
        uint4 hcw[kStreams];
        float inj[kStreams];
        auto const* hc_row = reinterpret_cast<uint8_t const*>(params.hc) + row * params.hc_stride * 2;
#pragma unroll
        for (int s = 0; s < kStreams; ++s)
        {
            hcw[s] = reinterpret_cast<uint4 const*>(hc_row + s * kHidden * 2)[tid];
            inj[s] = ld_bf16(params.inject, row * params.inject_stride + s);
        }
        [[maybe_unused]] uint4 shw = make_uint4(0, 0, 0, 0);
        [[maybe_unused]] float gate = 0.f;
        if constexpr (HasShared)
        {
            shw = reinterpret_cast<uint4 const*>(
                reinterpret_cast<uint8_t const*>(params.shared) + row * params.shared_stride * 2)[tid];
            gate = ld_bf16(params.gate, row * params.gate_stride);
        }

        float4 vals[NRanks];
        bool done = false;
        while (!done)
        {
            done = true;
#pragma unroll
            for (int r = 0; r < NRanks; ++r)
            {
                vals[r]
                    = ld_global_volatile(&reinterpret_cast<float4*>(comm.data_bufs[params.rank])[r * tot_access + idx]);
                done &= !is_neg_zero(vals[r]);
            }
        }
        // The all-reduce's bits: fp32 sum in rank order, rounded once.
        float b[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            float acc = static_cast<float>(reinterpret_cast<__nv_bfloat16 const*>(&vals[0])[i]);
#pragma unroll
            for (int r = 1; r < NRanks; ++r)
            {
                acc += static_cast<float>(reinterpret_cast<__nv_bfloat16 const*>(&vals[r])[i]);
            }
            b[i] = round_bf16(acc);
        }
        if constexpr (HasShared)
        {
            float const sg = sigmoid_triton(gate);
            uint32_t const* sw = reinterpret_cast<uint32_t const*>(&shw);
#pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                b[2 * p] = round_bf16(__fmaf_rn(sg, bf16_lo(sw[p]), b[2 * p]));
                b[2 * p + 1] = round_bf16(__fmaf_rn(sg, bf16_hi(sw[p]), b[2 * p + 1]));
            }
        }
        // Injection (rounded once, stored) and the per-thread sum-of-squares pairs.
        uint32_t ow[kStreams][4];
#pragma unroll
        for (int s = 0; s < kStreams; ++s)
        {
            uint32_t const* xw = reinterpret_cast<uint32_t const*>(&hcw[s]);
            float e[8];
#pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                ow[s][p] = pack_bf16x2(
                    __fmaf_rn(inj[s], b[2 * p], bf16_lo(xw[p])), __fmaf_rn(inj[s], b[2 * p + 1], bf16_hi(xw[p])));
                e[2 * p] = bf16_lo(ow[s][p]);
                e[2 * p + 1] = bf16_hi(ow[s][p]);
            }
            reinterpret_cast<uint4*>(
                reinterpret_cast<uint8_t*>(params.hc_out) + (row * params.hc_out_stride + s * kHidden) * 2)[tid]
                = make_uint4(ow[s][0], ow[s][1], ow[s][2], ow[s][3]);
            float const lo
                = __fadd_rn(__fmaf_rn(e[0], e[0], __fmul_rn(e[2], e[2])), __fmaf_rn(e[4], e[4], __fmul_rn(e[6], e[6])));
            float const hi
                = __fadd_rn(__fmaf_rn(e[1], e[1], __fmul_rn(e[3], e[3])), __fmaf_rn(e[5], e[5], __fmul_rn(e[7], e[7])));
            s_leaf[s][tid] = make_float2(lo, hi);
        }
        __syncthreads();
        if (tid < G)
        {
            float v[kStreams];
#pragma unroll
            for (int s = 0; s < kStreams; ++s)
            {
                v[s] = thread_ssq<G>(s_leaf[s], tid);
            }
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1)
            {
#pragma unroll
                for (int s = 0; s < kStreams; ++s)
                {
                    v[s] = __fadd_rn(v[s], __shfl_xor_sync(0xffffffffu, v[s], off));
                }
            }
            if (lane == 0)
            {
#pragma unroll
                for (int s = 0; s < kStreams; ++s)
                {
                    s_red[s][tid / 32] = v[s];
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int s = 0; s < kStreams; ++s)
        {
            float tot;
            if constexpr (G == 128)
            {
                tot = __fadd_rn(__fadd_rn(s_red[s][0], s_red[s][2]), __fadd_rn(s_red[s][1], s_red[s][3]));
            }
            else
            {
                tot = __fadd_rn(s_red[s][0], s_red[s][1]);
            }
            float const rstd = rstd_triton(tot, params.eps);
            uint4 const wq = s_w[s][tid];
            uint32_t const* ww = reinterpret_cast<uint32_t const*>(&wq);
            uint32_t nw[4];
#pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                float const ylo = __fmul_rn(__fmul_rn(bf16_lo(ow[s][p]), rstd), __fadd_rn(1.f, bf16_lo(ww[p])));
                float const yhi = __fmul_rn(__fmul_rn(bf16_hi(ow[s][p]), rstd), __fadd_rn(1.f, bf16_hi(ww[p])));
                nw[p] = pack_bf16x2(ylo, yhi);
            }
            reinterpret_cast<uint4*>(
                reinterpret_cast<uint8_t*>(params.normed) + (row * params.normed_stride + s * kHidden) * 2)[tid]
                = make_uint4(nw[0], nw[1], nw[2], nw[3]);
        }
    }

    comm.update(static_cast<int64_t>(rows) * kHidden * NRanks * sizeof(__nv_bfloat16));

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

template <int NRanks, int G, bool HasShared, bool TriggerCompletionAtEnd, int Var>
int max_grid()
{
    static int const grid = []()
    {
        int per_sm = 0;
        TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, hc_allreduce_inject_norm_kernel<NRanks, G, HasShared, TriggerCompletionAtEnd, Var>, kThreads, 0));
        return std::max(1, per_sm) * get_sm_count();
    }();
    return grid;
}

template <int NRanks, int G, bool HasShared, bool TriggerCompletionAtEnd, int Var>
void launch_var(HcInjectNormParams const& params)
{
    auto kernel = hc_allreduce_inject_norm_kernel<NRanks, G, HasShared, TriggerCompletionAtEnd, Var>;
    int const max_grid = hc_inject::max_grid<NRanks, G, HasShared, TriggerCompletionAtEnd, Var>();
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attribute[1];
    cfg.gridDim = std::min(params.rows, max_grid);
    cfg.blockDim = kThreads;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = params.stream;
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cfg.attrs = attribute;
    cfg.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, params));
}

template <int NRanks, int G, bool HasShared, bool TriggerCompletionAtEnd>
void launch(HcInjectNormParams const& params)
{
    int const g3 = hc_inject::max_grid<NRanks, G, HasShared, TriggerCompletionAtEnd, 3>();
    int const g2 = hc_inject::max_grid<NRanks, G, HasShared, TriggerCompletionAtEnd, 2>();
    if (params.rows > g3 && params.rows <= g2)
    {
        launch_var<NRanks, G, HasShared, TriggerCompletionAtEnd, 2>(params);
    }
    else
    {
        launch_var<NRanks, G, HasShared, TriggerCompletionAtEnd, 3>(params);
    }
}

template <int NRanks, int G>
void dispatch_flags(HcInjectNormParams const& params)
{
    bool const shared = params.shared != nullptr;
    if (shared && params.trigger_completion_at_end)
    {
        launch<NRanks, G, true, true>(params);
    }
    else if (shared)
    {
        launch<NRanks, G, true, false>(params);
    }
    else if (params.trigger_completion_at_end)
    {
        launch<NRanks, G, false, true>(params);
    }
    else
    {
        launch<NRanks, G, false, false>(params);
    }
}
} // namespace hc_inject

void hc_allreduce_inject_norm_op(HcInjectNormParams const& params)
{
    TLLM_CHECK_WITH_INFO(params.hidden_dim == hc_inject::kHidden && params.hc_count == hc_inject::kStreams,
        "hc_allreduce_inject_norm: hidden 2560 and 4 streams only");
    TLLM_CHECK_WITH_INFO(params.rows > 0, "hc_allreduce_inject_norm: no rows");
    TLLM_CHECK_WITH_INFO(params.program_warps == 4 || params.program_warps == 2,
        "hc_allreduce_inject_norm: the Triton program's warps must be 4 or 2");
    TLLM_CHECK_WITH_INFO((params.shared == nullptr) == (params.gate == nullptr),
        "hc_allreduce_inject_norm: shared and gate come together");
    TLLM_CHECK_WITH_INFO(params.partial != nullptr, "hc_allreduce_inject_norm: no partial");
    if (params.nranks == 2)
    {
        params.program_warps == 4 ? hc_inject::dispatch_flags<2, 128>(params)
                                  : hc_inject::dispatch_flags<2, 64>(params);
    }
    else if (params.nranks == 4)
    {
        params.program_warps == 4 ? hc_inject::dispatch_flags<4, 128>(params)
                                  : hc_inject::dispatch_flags<4, 64>(params);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "hc_allreduce_inject_norm: 2 or 4 ranks only");
    }
}
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
