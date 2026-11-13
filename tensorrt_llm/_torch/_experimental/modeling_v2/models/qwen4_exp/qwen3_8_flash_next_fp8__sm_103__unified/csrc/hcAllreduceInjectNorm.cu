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
// This kernel shares the stock IPC all-reduce workspace with the Lamport kernels of
// communicationKernels/allReduceFusionKernels.cu and runs between their calls, so LamportComm, the helpers below and
// the push / clear / poll / update steps must follow that file's Lamport protocol:
//   - workspace[2 * NRanks + r] is rank r's buffer of three sub-buffers of workspace[3 * NRanks + 1][1] bytes; with
//     flag = workspace[3 * NRanks][2], a call writes and polls sub-buffer flag % 3 and clears its own (flag + 2) % 3.
//   - workspace[3 * NRanks + 1][0] holds the bytes the previous call wrote, cleared as 16-byte stores of -0.0f.
//   - A 16-byte slot has arrived once none of its 32-bit words is -0.0f, so senders store -0.0f as +0.0f.
//   - Every CTA increments workspace[3 * NRanks][0] after reading the flag; once all have, CTA 0 advances the flag to
//     (flag + 1) % 3, stores the bytes this call wrote and resets the counter.
// The comm op matrix (tests/unittest/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified/comm/
// _hc_ar_inject_norm_op_matrix.py) pins this: it interleaves this kernel with stock all-reduces on one workspace and
// checks both bitwise at 2 and 4 ranks.
template <int NRanks>
struct LamportComm
{
    __device__ __forceinline__ LamportComm(void** workspace, int rank)
    {
        counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
        flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
        // Bytes the previous Lamport kernel wrote to its sub-buffer. Every Lamport kernel on this workspace reads and
        // writes it, whatever its dtype, so it stays in bytes. Readers round it up to whole 16-byte stores: the scalar
        // MiniMax kernel writes 4-byte words.
        clear_ptr = &reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[0];
        flag_value = *flag_ptr;
        auto comm_size = reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[1];
        clear_size = *clear_ptr;
        int data_offset = flag_value % 3;
        int clear_offset = (flag_value + 2) % 3;
        for (int r = 0; r < NRanks; ++r)
        {
            data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) + data_offset * comm_size;
        }
        clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int64_t new_clear_size)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x)
            {
            }
            *flag_ptr = (flag_value + 1) % 3;
            *clear_ptr = new_clear_size;
            *counter_ptr = 0;
        }
    }

    int* counter_ptr;
    int* flag_ptr;
    int64_t* clear_ptr;
    uint8_t* data_bufs[NRanks];
    uint8_t* clear_buf;
    int64_t clear_size;
    int flag_value;
};

//! -0.0f is the not-arrived marker: a 16-byte slot has arrived once none of its words is -0.0f.
__device__ __forceinline__ bool is_neg_zero(float v)
{
    return *reinterpret_cast<uint32_t*>(&v) == 0x80000000;
}

__device__ __forceinline__ bool is_neg_zero(float4 v)
{
    return is_neg_zero(v.x) || is_neg_zero(v.y) || is_neg_zero(v.z) || is_neg_zero(v.w);
}

__device__ __forceinline__ float4 get_neg_zero()
{
    float4 vec;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000;
    }
    return vec;
}

__device__ __forceinline__ float4 ld_global_volatile(float4* addr)
{
    float4 val;
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(addr));
    return val;
}

//! The current device's SM count, read once per process.
inline int get_sm_count()
{
    static int sm_count = 0;
    if (sm_count == 0)
    {
        int device_id;
        TLLM_CUDA_CHECK(cudaGetDevice(&device_id));
        cudaDeviceProp device_prop;
        TLLM_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
        sm_count = device_prop.multiProcessorCount;
    }
    return sm_count;
}
} // namespace hc_inject

// Thread t holds columns [8t, 8t + 8) of a row's block output and of each of its 4 streams. The Triton program runs a
// stream on G = 32 * programWarps threads over a 4096-wide block, its thread u in chunk c holding this CTA's thread
// c * G + u. Its sum of squares, reproduced bitwise: per thread and chunk the fp32 pair (fma(x0, x0, x2 * x2) +
// fma(x4, x4, x6 * x6), fma(x1, x1, x3 * x3) + fma(x5, x5, x7 * x7)), a pairwise tree over chunks (masked ones are
// zero), low + high, the warp's xor butterfly 16..1, then ((w0 + w2) + (w1 + w3)) at 4 warps or (w0 + w1) at 2.
namespace hc_inject
{
static constexpr int kHidden = 2560;
static constexpr int kStreams = 4;
static constexpr int kThreads = kHidden / 8;

__device__ __forceinline__ float bf16Lo(uint32_t w)
{
    return __uint_as_float(w << 16);
}

__device__ __forceinline__ float bf16Hi(uint32_t w)
{
    return __uint_as_float(w & 0xFFFF0000u);
}

__device__ __forceinline__ uint32_t packBf16x2(float lo, float hi)
{
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ float roundBf16(float x)
{
    return bf16Lo(packBf16x2(x, 0.f));
}

__device__ __forceinline__ float ldBf16(void const* base, int64_t elem)
{
    uint16_t v = reinterpret_cast<uint16_t const*>(base)[elem];
    return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

//! `tl.sigmoid` as the Triton kernel compiles it.
__device__ __forceinline__ float sigmoidTriton(float x)
{
    float y;
    asm("{ .reg .f32 t; mul.f32 t, %1, 0fBFB8AA3B; ex2.approx.f32 t, t; add.f32 t, t, 0f3F800000; "
        "div.full.f32 %0, 0f3F800000, t; }"
        : "=f"(y)
        : "f"(x));
    return y;
}

//! `rsqrt(ss / 2560 + eps)` as the Triton kernel compiles it (`div.full`, `add`, `rsqrt.approx.ftz`).
__device__ __forceinline__ float rstdTriton(float ss, float eps)
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
__device__ __forceinline__ float threadSsq(float2 const* leaves, int u)
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
    hcAllreduceInjectNormKernel(HcInjectNormParams params)
{
    static_assert(kHidden % (8 * G) == 0 || G == 128, "the chunk layout above");
    __shared__ float2 sLeaf[kStreams][kThreads];
    __shared__ float sRed[kStreams][G / 32];
    __shared__ uint4 sW[kStreams][kThreads];
    int const tid = threadIdx.x;
    int const lane = tid & 31;

    // No kernel of the step writes the norm weights: stage them before the PDL wait (each thread reads back its own).
#pragma unroll
    for (int s = 0; s < kStreams; ++s)
    {
        sW[s][tid] = reinterpret_cast<uint4 const*>(params.normWeight)[s * kThreads + tid];
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
    int const totAccess = rows * kThreads;
    int const clearAccess = static_cast<int>((comm.clear_size + sizeof(float4) - 1) / sizeof(float4));

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
            reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * totAccess + idx]
                = *reinterpret_cast<float4*>(val);
        }
    }
    float4 const clearVec = get_neg_zero();
    for (int idx = blockIdx.x * kThreads + tid; idx < clearAccess; idx += gridDim.x * kThreads)
    {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clearVec;
    }

    for (int row = blockIdx.x; row < rows; row += gridDim.x)
    {
        int const idx = row * kThreads + tid;
        // The site's operands (issued before the poll, consumed after it).
        uint4 hcw[kStreams];
        float inj[kStreams];
        auto const* hcRow = reinterpret_cast<uint8_t const*>(params.hc) + row * params.hcStride * 2;
#pragma unroll
        for (int s = 0; s < kStreams; ++s)
        {
            hcw[s] = reinterpret_cast<uint4 const*>(hcRow + s * kHidden * 2)[tid];
            inj[s] = ldBf16(params.inject, row * params.injectStride + s);
        }
        [[maybe_unused]] uint4 shw = make_uint4(0, 0, 0, 0);
        [[maybe_unused]] float gate = 0.f;
        if constexpr (HasShared)
        {
            shw = reinterpret_cast<uint4 const*>(
                reinterpret_cast<uint8_t const*>(params.shared) + row * params.sharedStride * 2)[tid];
            gate = ldBf16(params.gate, row * params.gateStride);
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
                    = ld_global_volatile(&reinterpret_cast<float4*>(comm.data_bufs[params.rank])[r * totAccess + idx]);
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
            b[i] = roundBf16(acc);
        }
        if constexpr (HasShared)
        {
            float const sg = sigmoidTriton(gate);
            uint32_t const* sw = reinterpret_cast<uint32_t const*>(&shw);
#pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                b[2 * p] = roundBf16(__fmaf_rn(sg, bf16Lo(sw[p]), b[2 * p]));
                b[2 * p + 1] = roundBf16(__fmaf_rn(sg, bf16Hi(sw[p]), b[2 * p + 1]));
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
                ow[s][p] = packBf16x2(
                    __fmaf_rn(inj[s], b[2 * p], bf16Lo(xw[p])), __fmaf_rn(inj[s], b[2 * p + 1], bf16Hi(xw[p])));
                e[2 * p] = bf16Lo(ow[s][p]);
                e[2 * p + 1] = bf16Hi(ow[s][p]);
            }
            reinterpret_cast<uint4*>(
                reinterpret_cast<uint8_t*>(params.hcOut) + (row * params.hcOutStride + s * kHidden) * 2)[tid]
                = make_uint4(ow[s][0], ow[s][1], ow[s][2], ow[s][3]);
            float const lo
                = __fadd_rn(__fmaf_rn(e[0], e[0], __fmul_rn(e[2], e[2])), __fmaf_rn(e[4], e[4], __fmul_rn(e[6], e[6])));
            float const hi
                = __fadd_rn(__fmaf_rn(e[1], e[1], __fmul_rn(e[3], e[3])), __fmaf_rn(e[5], e[5], __fmul_rn(e[7], e[7])));
            sLeaf[s][tid] = make_float2(lo, hi);
        }
        __syncthreads();
        if (tid < G)
        {
            float v[kStreams];
#pragma unroll
            for (int s = 0; s < kStreams; ++s)
            {
                v[s] = threadSsq<G>(sLeaf[s], tid);
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
                    sRed[s][tid / 32] = v[s];
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
                tot = __fadd_rn(__fadd_rn(sRed[s][0], sRed[s][2]), __fadd_rn(sRed[s][1], sRed[s][3]));
            }
            else
            {
                tot = __fadd_rn(sRed[s][0], sRed[s][1]);
            }
            float const rstd = rstdTriton(tot, params.eps);
            uint4 const wq = sW[s][tid];
            uint32_t const* ww = reinterpret_cast<uint32_t const*>(&wq);
            uint32_t nw[4];
#pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                float const ylo = __fmul_rn(__fmul_rn(bf16Lo(ow[s][p]), rstd), __fadd_rn(1.f, bf16Lo(ww[p])));
                float const yhi = __fmul_rn(__fmul_rn(bf16Hi(ow[s][p]), rstd), __fadd_rn(1.f, bf16Hi(ww[p])));
                nw[p] = packBf16x2(ylo, yhi);
            }
            reinterpret_cast<uint4*>(
                reinterpret_cast<uint8_t*>(params.normed) + (row * params.normedStride + s * kHidden) * 2)[tid]
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
int maxGrid()
{
    static int const grid = []()
    {
        int perSm = 0;
        TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &perSm, hcAllreduceInjectNormKernel<NRanks, G, HasShared, TriggerCompletionAtEnd, Var>, kThreads, 0));
        return std::max(1, perSm) * get_sm_count();
    }();
    return grid;
}

template <int NRanks, int G, bool HasShared, bool TriggerCompletionAtEnd, int Var>
void launchVar(HcInjectNormParams const& params)
{
    auto kernel = hcAllreduceInjectNormKernel<NRanks, G, HasShared, TriggerCompletionAtEnd, Var>;
    int const maxGrid = hc_inject::maxGrid<NRanks, G, HasShared, TriggerCompletionAtEnd, Var>();
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attribute[1];
    cfg.gridDim = std::min(params.rows, maxGrid);
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
    int const g3 = hc_inject::maxGrid<NRanks, G, HasShared, TriggerCompletionAtEnd, 3>();
    int const g2 = hc_inject::maxGrid<NRanks, G, HasShared, TriggerCompletionAtEnd, 2>();
    if (params.rows > g3 && params.rows <= g2)
    {
        launchVar<NRanks, G, HasShared, TriggerCompletionAtEnd, 2>(params);
    }
    else
    {
        launchVar<NRanks, G, HasShared, TriggerCompletionAtEnd, 3>(params);
    }
}

template <int NRanks, int G>
void dispatchFlags(HcInjectNormParams const& params)
{
    bool const shared = params.shared != nullptr;
    if (shared && params.triggerCompletionAtEnd)
    {
        launch<NRanks, G, true, true>(params);
    }
    else if (shared)
    {
        launch<NRanks, G, true, false>(params);
    }
    else if (params.triggerCompletionAtEnd)
    {
        launch<NRanks, G, false, true>(params);
    }
    else
    {
        launch<NRanks, G, false, false>(params);
    }
}
} // namespace hc_inject

void launchHcAllreduceInjectNorm(HcInjectNormParams const& params)
{
    TLLM_CHECK_WITH_INFO(params.hiddenDim == hc_inject::kHidden && params.hcCount == hc_inject::kStreams,
        "hc_allreduce_inject_norm: hidden 2560 and 4 streams only");
    TLLM_CHECK_WITH_INFO(params.rows > 0, "hc_allreduce_inject_norm: no rows");
    TLLM_CHECK_WITH_INFO(params.programWarps == 4 || params.programWarps == 2,
        "hc_allreduce_inject_norm: the Triton program's warps must be 4 or 2");
    TLLM_CHECK_WITH_INFO((params.shared == nullptr) == (params.gate == nullptr),
        "hc_allreduce_inject_norm: shared and gate come together");
    TLLM_CHECK_WITH_INFO(params.partial != nullptr, "hc_allreduce_inject_norm: no partial");
    if (params.nranks == 2)
    {
        params.programWarps == 4 ? hc_inject::dispatchFlags<2, 128>(params) : hc_inject::dispatchFlags<2, 64>(params);
    }
    else if (params.nranks == 4)
    {
        params.programWarps == 4 ? hc_inject::dispatchFlags<4, 128>(params) : hc_inject::dispatchFlags<4, 64>(params);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "hc_allreduce_inject_norm: 2 or 4 ranks only");
    }
}
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
