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

// NATIVE_ADD_RMSNORM (bitwise vs the served kernels): see addRmsNormSm107.h. The sum of squares keeps
// FlashInfer's FusedAddRMSNormKernel order for H 6144 (64 reference lanes per row; lane l owns vectors l + 64 b,
// b = 0..11; an in-thread fma chain over its 96 values, then a butterfly 1..16 per warp, then warp 0 + warp 1).
// Geometry: a cluster of C CTAs per row; CTA c owns the reference lanes [c L, (c + 1) L), L = 64 / C, and S threads per
// lane split its 12 vector blocks (S > 1 hands the h values to the lane's first thread through shared memory). The
// cross-CTA step of the butterfly goes through distributed shared memory (one cluster barrier).
#include "minimax_m3/csrc/m31/addRmsNormSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{
namespace cg = cooperative_groups;

constexpr int kH = kAddRmsNormHidden;
constexpr int kVec = 8;
constexpr int kRefLanes = 64;
constexpr int kBlocks = kH / kVec / kRefLanes; // 12 vector blocks per reference lane
constexpr int kKTiles = (kH / 32 + 3) / 4;     // 48 scale words per row
constexpr uint32_t kPrefetchChunk = 16384;     // bytes per cp.async.bulk.prefetch.L2

// NORM_L2PF: CTA `issuer` of `issuers` pulls its 4 KiB-rounded share of [base, base + bytes) into L2, its
// threads striding over 16 KiB chunks (the prototype's split).
__device__ __forceinline__ void prefetchL2Slice(
    char const* base, uint64_t bytes, int issuer, int issuers, int tid, int threads)
{
    if (base == nullptr || bytes == 0)
    {
        return;
    }
    uint64_t const per = (bytes / issuers + 4095) & ~uint64_t(4095);
    uint64_t const beg = per * issuer;
    uint64_t const end = beg + per < bytes ? beg + per : bytes;
    for (uint64_t off = beg + uint64_t(tid) * kPrefetchChunk; off < end; off += uint64_t(threads) * kPrefetchChunk)
    {
        uint32_t const size = static_cast<uint32_t>((end - off) < kPrefetchChunk ? (end - off) : kPrefetchChunk) & ~15u;
        if (size != 0)
        {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + off), "r"(size) : "memory");
        }
    }
}

__device__ __forceinline__ float rsqrtApprox(float x)
{
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float rcpApproxFtz(float x)
{
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float divFull(float a, float b)
{
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ uint32_t ue8m0Rp(float x)
{
    uint32_t r;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)                                        \
    && (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
    asm("{\n\t.reg .b16 t;\n\tcvt.rp.satfinite.ue8m0x2.f32 t, 0f00000000, %1;\n\tcvt.u32.u16 %0, t;\n\t}\n"
        : "=r"(r)
        : "f"(x));
#else
    r = 0u;
    __trap(); // the host dispatches the MXFP8 forms only to sm_107a / sm_10xf builds
#endif
    return r & 0xFFu;
}

__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi)
{
    uint32_t r;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    asm("{\n\t.reg .b16 t;\n\tcvt.rn.satfinite.e4m3x2.f32 t, %1, %2;\n\tcvt.u32.u16 %0, t;\n\t}\n"
        : "=r"(r)
        : "f"(hi), "f"(lo));
#else
    r = 0u;
    __trap();
#endif
    return r;
}

__device__ __forceinline__ void unpack(uint4 u, float f[8])
{
    __nv_bfloat162 const* p = reinterpret_cast<__nv_bfloat162 const*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        f[2 * i] = __low2float(p[i]);
        f[2 * i + 1] = __high2float(p[i]);
    }
}

__device__ __forceinline__ uint4 pack(float const f[8])
{
    uint4 u;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        p[i] = __halves2bfloat162(__float2bfloat16_rn(f[2 * i]), __float2bfloat16_rn(f[2 * i + 1]));
    return u;
}

//! The in-thread sum of squares over the lane's 96 values in vector-block order (value e of block b = index 8 b + e):
//! FlashInfer's fma chain.
__device__ __forceinline__ float laneSumSq(float const (&x)[kBlocks * kVec])
{
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < kBlocks * kVec; ++i)
        acc = __fmaf_rn(x[i], x[i], acc);
    return acc;
}

//! The MXFP8 form of one 8-element vector (vector v of row): mxfp8_rows.store_mxfp8_vector. Every lane of the warp
//! calls it for its vector of the same block slot.
__device__ __forceinline__ void storeMxfp8Vector(
    float const x[8], uint64_t* q, uint32_t* sfWords, int64_t row, int v, float rcp448, int lane)
{
    float amax = fabsf(x[0]);
#pragma unroll
    for (int j = 1; j < 8; ++j)
        amax = fmaxf(amax, fabsf(x[j]));
#pragma unroll
    for (int step = 0; step < 2; ++step)
        amax = fmaxf(__shfl_xor_sync(0xFFFFFFFFu, amax, 1 << step), amax);
    uint32_t const scale = ue8m0Rp(__fmul_rn(amax, rcp448));
    float const outScale = scale != 0u ? rcpApproxFtz(__uint_as_float(scale << 23)) : 0.0f;
    uint32_t const w0 = e4m3x2(__fmul_rn(x[0], outScale), __fmul_rn(x[1], outScale))
        | (e4m3x2(__fmul_rn(x[2], outScale), __fmul_rn(x[3], outScale)) << 16);
    uint32_t const w1 = e4m3x2(__fmul_rn(x[4], outScale), __fmul_rn(x[5], outScale))
        | (e4m3x2(__fmul_rn(x[6], outScale), __fmul_rn(x[7], outScale)) << 16);
    q[row * (kH / kVec) + v] = (static_cast<uint64_t>(w1) << 32) | static_cast<uint64_t>(w0);
    uint32_t const s1 = __shfl_down_sync(0xFFFFFFFFu, scale, 4);
    uint32_t const s2 = __shfl_down_sync(0xFFFFFFFFu, scale, 8);
    uint32_t const s3 = __shfl_down_sync(0xFFFFFFFFu, scale, 12);
    if ((lane & 15) == 0)
    {
        int const kTile = v / 16;
        int64_t const word = (row / 128) * (kKTiles * 128) + kTile * 128 + (row % 32) * 4 + (row % 128) / 32;
        sfWords[word] = scale | (s1 << 8) | (s2 << 16) | (s3 << 24);
    }
}

template <int C, int S>
__global__ void __launch_bounds__(kRefLanes / C * S, 1) addRmsNormKernel(AddRmsNormParams p, bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int L = kRefLanes / C;
    constexpr int T = L * S;
    constexpr int BPS = kBlocks / S; // vector blocks per thread
    static_assert(kBlocks % S == 0, "S must divide 12");
    __shared__ __align__(16) float sH[S > 1 ? kBlocks * L * kVec : 1];
    __shared__ float sPart[C > 1 ? C : 2];
    int const tid = threadIdx.x;
    int const s = tid / L, l = tid % L, lane = tid & 31;
    int const c = C > 1 ? static_cast<int>(cg::this_cluster().block_rank()) : 0;
    int64_t const row = blockIdx.y;
    int const refLane = c * L + l;
    if (p.prefetch0Bytes != 0 || p.prefetch1Bytes != 0)
    {
        int const cta = static_cast<int>(row) * C + c;
        int const ctas = p.numRows * C;
        prefetchL2Slice(p.prefetch0, p.prefetch0Bytes, cta, ctas, tid, T);
        prefetchL2Slice(p.prefetch1, p.prefetch1Bytes, cta, ctas, tid, T);
    }

    uint4 wv[BPS];
#pragma unroll
    for (int i = 0; i < BPS; ++i)
        wv[i] = *reinterpret_cast<uint4 const*>(p.weight + (refLane + kRefLanes * (s * BPS + i)) * kVec);
    if (pdl)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    __nv_bfloat16* inRow = p.input + row * kH;
    __nv_bfloat16* resRow = p.residual + row * kH;
    uint4 iv[BPS], rv[BPS];
#pragma unroll
    for (int i = 0; i < BPS; ++i)
    {
        int const v = refLane + kRefLanes * (s * BPS + i);
        iv[i] = *reinterpret_cast<uint4 const*>(inRow + v * kVec);
        rv[i] = *reinterpret_cast<uint4 const*>(resRow + v * kVec);
    }
    float h[BPS][kVec];
#pragma unroll
    for (int i = 0; i < BPS; ++i)
    {
        int const v = refLane + kRefLanes * (s * BPS + i);
        float a[8], b[8];
        unpack(iv[i], a);
        unpack(rv[i], b);
#pragma unroll
        for (int e = 0; e < kVec; ++e)
            h[i][e] = __fadd_rn(a[e], b[e]);
        *reinterpret_cast<uint4*>(resRow + v * kVec) = pack(h[i]);
        if (S > 1 && s != 0)
#pragma unroll
            for (int e = 0; e < kVec; ++e)
                sH[((s * BPS + i) * L + l) * kVec + e] = h[i][e];
    }
    if (S > 1)
        __syncthreads();
    // The lane's sum of squares, by its first thread (s == 0), then the butterfly across the lanes of this CTA.
    float part = 0.0f;
    if (tid < (L < 32 ? 32 : L))
    {
        if (s == 0)
        {
            float x[kBlocks * kVec];
#pragma unroll
            for (int b = 0; b < kBlocks; ++b)
#pragma unroll
                for (int e = 0; e < kVec; ++e)
                    x[b * kVec + e] = b < BPS ? h[b % BPS][e] : sH[(b * L + l) * kVec + e];
            part = laneSumSq(x);
        }
#pragma unroll
        for (int o = 1; o < (L < 32 ? L : 32); o <<= 1)
            part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, o));
    }
    float sum;
    if constexpr (C == 1)
    {
        // Two reference warps in this CTA: warp 0 + warp 1.
        if (tid < 64 && lane == 0)
            sPart[tid / 32] = part;
        __syncthreads();
        sum = __fadd_rn(sPart[0], sPart[1]);
    }
    else
    {
        cg::cluster_group cluster = cg::this_cluster();
        if (tid == 0)
            for (int r = 0; r < C; ++r)
                *cluster.map_shared_rank(&sPart[c], r) = part;
        cluster.sync();
        if constexpr (C == 2)
            sum = __fadd_rn(sPart[0], sPart[1]);
        else
            sum = __fadd_rn(__fadd_rn(sPart[0], sPart[1]), __fadd_rn(sPart[2], sPart[3]));
    }
    float const mean = __fdiv_rn(sum, static_cast<float>(kH));
    float const rstd = rsqrtApprox(__fadd_rn(mean, p.eps));
    float const rcp448 = rcpApproxFtz(448.0f);
#pragma unroll
    for (int i = 0; i < BPS; ++i)
    {
        int const v = refLane + kRefLanes * (s * BPS + i);
        float w[8], y[8], yb[8];
        unpack(wv[i], w);
#pragma unroll
        for (int e = 0; e < kVec; ++e)
            y[e] = __fmul_rn(__fmul_rn(h[i][e], rstd), __fadd_rn(w[e], 0.0f));
        uint4 const out = pack(y);
        *reinterpret_cast<uint4*>(inRow + v * kVec) = out;
        for (int r = 0; r < p.numReplicas; ++r)
            *reinterpret_cast<uint4*>(p.replicas + (row * p.numReplicas + r) * kH + v * kVec) = out;
        if (p.q != nullptr)
        {
            unpack(out, yb);
            storeMxfp8Vector(yb, p.q, p.sfWords, row, v, rcp448, lane);
        }
    }
    if (p.q != nullptr)
    {
        // The scale words of the rows padding numRows to a multiple of 128 (all in the last tile) are zero.
        int const lastTile = p.numRows / 128, rowsInTile = p.numRows - lastTile * 128;
        if (rowsInTile != 0)
        {
            int64_t const base = static_cast<int64_t>(lastTile) * kKTiles * 128;
            int const total = p.numRows * C * T;
            for (int word = (static_cast<int>(row) * C + c) * T + tid; word < kKTiles * 128; word += total)
            {
                int const w = word % 128;
                if ((w % 4) * 32 + w / 4 >= rowsInTile)
                    p.sfWords[base + word] = 0u;
            }
        }
    }
#endif
}

template <int C, int S>
cudaError_t launch(AddRmsNormParams const& p, bool pdl, cudaStream_t stream)
{
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(C, static_cast<unsigned>(p.numRows));
    cfg.blockDim = dim3(kRefLanes / C * S);
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    int n = 0;
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    ++n;
    if (C > 1)
    {
        attr[n].id = cudaLaunchAttributeClusterDimension;
        attr[n].val.clusterDim.x = C;
        attr[n].val.clusterDim.y = 1;
        attr[n].val.clusterDim.z = 1;
        ++n;
    }
    cfg.attrs = attr;
    cfg.numAttrs = n;
    return cudaLaunchKernelEx(&cfg, addRmsNormKernel<C, S>, p, pdl);
}

} // namespace

cudaError_t invokeAddRmsNorm(AddRmsNormParams const& params, bool pdl, cudaStream_t stream)
{
    if (params.numRows <= 0)
    {
        return cudaSuccess;
    }
    if (params.numRows > 65535 || (params.q == nullptr) != (params.sfWords == nullptr) || params.numReplicas < 0)
    {
        return cudaErrorInvalidValue;
    }
    // Dispatch table (every (form, rows) cell faster than the served kernel): a 4-CTA cluster x 12 threads
    // per lane up to 92 rows, a 2-CTA cluster x 12 up to 106, then one CTA x 4 threads per lane (MXFP8 and plain
    // forms) or x 3 (BF16 replicas only).
    if (params.numRows <= 92)
    {
        return launch<4, 12>(params, pdl, stream);
    }
    if (params.numRows <= 106)
    {
        return launch<2, 12>(params, pdl, stream);
    }
    if (params.q == nullptr && params.numReplicas > 0)
    {
        return launch<1, 3>(params, pdl, stream);
    }
    return launch<1, 4>(params, pdl, stream);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
