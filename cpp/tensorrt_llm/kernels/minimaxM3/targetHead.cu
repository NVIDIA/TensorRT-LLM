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

#include "tensorrt_llm/kernels/minimaxM3/targetHead.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// The MiniMax-M3 target lm_head shard GEMV of a verify step of at most 8 rows (GB300, TP4), at stock precision: BF16
// weight and activation, mma.m16n8k16 BF16 with FP32 accumulation over the full K, one rounding to BF16 per logit
// (the cuBLAS head differs only in the FP32 summation order). The full logit row is produced.
//
// * Lossless 12-bit weight coding: the BF16 exponents of a trained weight sit almost all in a window of 15
//   consecutive values [base, base + 14]. Each weight is one byte sign << 7 | mantissa plus a 4-bit code
//   exponent - base; the decoded BF16 bits are the checkpoint's. Code 15 marks an exception: it decodes to +0 (it
//   adds nothing to the MMA sums), and the segment epilogue adds the exact product x * w from a per-tile list (a
//   product of two BF16 values is exact in FP32), so every product enters the sum exactly as in the BF16 head.
// * Layout: the weight is permuted into mma A-fragment order (per 16-row tile and k16 step, lane (g, j) holds
//   {row g k 2j..2j+1, row g+8 k 2j.., row g k 2j+8.., row g+8 k 2j+8..}); per unit of 4 k16 steps a lane reads
//   48 bytes: the sign/mantissa bytes of steps 0-1 and of steps 2-3, then 4 code words (even values' nibbles in the
//   low half, odd values' in the high half, so the codes of value pair p are (word >> 4p) & 0x000f000f).
// * A persistent grid (one 16-warp CTA per SM) splits the (tile, unit) space into equal contiguous ranges, one per
//   warp, each streamed through a 3-unit cp.async ring in shared memory. The first units are issued before the PDL
//   grid-dependency wait (the weight does not depend on the previous kernel); the activation is staged in shared
//   memory in B-fragment order after it.
// * A tile split between warps writes per-warp FP32 partials; the last finisher (per-tile counter) adds them in warp
//   order, so the result is deterministic run to run.

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
namespace targetHead
{
namespace
{

constexpr int kUnit = 4; // k16 steps per coded unit

struct Args
{
    Params p;
    int tiles;
    int ks;          // K / 16
    long long units; // tiles * ks / kUnit
};

__device__ __forceinline__ void cp_async16(void* smem, void const* gmem)
{
    uint32_t const sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem) : "memory");
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint4 const& a, uint2 const& b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}

// Accumulator layout of one tile: c[0..1] = (row g, tokens 2j, 2j + 1), c[2..3] = (row g + 8, ...).
__device__ __forceinline__ void store_tile(Args const& a, int tile, int lane, float const (&c)[4])
{
    int const g = lane >> 2, j = lane & 3;
#pragma unroll
    for (int e = 0; e < 4; ++e)
    {
        int const n = 2 * j + (e & 1);
        int const row = tile * 16 + g + (e >> 1) * 8;
        if (n < a.p.M && row < a.p.N)
            a.p.out[n * a.p.out_stride + row] = __float2bfloat16_rn(c[e]);
    }
}

// Flush the warp's sums of `tile` over k16 steps [k0, k1).
__device__ __forceinline__ void flush(Args const& a, int warp, int tile, int k0, int k1, int lane, float const (&c)[4])
{
    if (k0 == 0 && k1 == a.ks)
    {
        store_tile(a, tile, lane, c);
        return;
    }
    // Warps owning parts of this tile: the owner of its first unit up to the owner of its last one, where
    // owner(u) = the largest w with w * units / W <= u. The first owner stores into its slot 1 (the tile is its last
    // segment), every later owner into slot 0 (its first segment), so a warp never reuses a slot.
    long long const W = static_cast<long long>(gridDim.x) * (blockDim.x >> 5);
    long long const first = static_cast<long long>(tile) * (a.ks / kUnit);
    auto owner = [&](long long u)
    {
        long long w = (u * W) / a.units;
        while (w + 1 < W && ((w + 1) * a.units) / W <= u)
            ++w;
        while (w > 0 && (w * a.units) / W > u)
            --w;
        return static_cast<int>(w);
    };
    int const wf = owner(first);
    int const wl = owner(first + a.ks / kUnit - 1);
    constexpr int kVals = 4 * 32;
    float* mine = a.p.part + (static_cast<long long>(warp) * 2 + (warp == wf ? 1 : 0)) * kVals;
#pragma unroll
    for (int e = 0; e < 4; ++e)
        mine[e * 32 + lane] = c[e];
    __threadfence();
    __syncwarp();
    int done = 0;
    if (lane == 0)
        done = atomicAdd(&a.p.counter[tile], 1) == wl - wf;
    done = __shfl_sync(0xffffffffu, done, 0);
    if (!done)
        return;
    __threadfence();
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    for (int w = wf; w <= wl; ++w)
    {
        float const* q = a.p.part + (static_cast<long long>(w) * 2 + (w == wf ? 1 : 0)) * kVals;
#pragma unroll
        for (int e = 0; e < 4; ++e)
            s[e] += __ldcg(q + e * 32 + lane);
    }
    store_tile(a, tile, lane, s);
    if (lane == 0)
        a.p.counter[tile] = 0;
}

// BF16 pair p (values 2p, 2p + 1) of a lane's 8 coded values: sign/mantissa bytes sm (4 per word), codes word cw.
__device__ __forceinline__ uint32_t decode_pair(uint32_t sm_word, int byte0, uint32_t cw, int p, uint32_t base)
{
    uint32_t const b = __byte_perm(sm_word, 0u, byte0 == 0 ? 0x4140 : 0x4342); // {b0, 0, b1, 0}
    uint32_t const q = (cw >> (4 * p)) & 0x000f000fu;
    uint32_t const e = q * 128u + base;                                        // exponent field of both halves
    // Code 15 (an exception, sign/mantissa byte 0) decodes to +0.
    return ((b & 0x007f007fu) | ((b << 8) & 0x80008000u) | e) & __vcmpne2(q, 0x000f000fu);
}

// WARPS warps per CTA, each with a cp.async ring of S units (1536 bytes each). The range is walked tile segment by
// tile segment, so the inner loop has no boundary checks; k16 steps alternate between two accumulators (added at the
// segment end). x is staged as [ks / 4][32][4] B fragments, two 16-byte shared loads per unit.
template <int WARPS, int S>
__global__ void __launch_bounds__(WARPS * 32, 1) coded_kernel(Args const a)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    extern __shared__ uint4 smem4[];
    int const lane = threadIdx.x & 31;
    int const wid = threadIdx.x >> 5;
    int const warp = blockIdx.x * WARPS + wid;
    long long const W = static_cast<long long>(gridDim.x) * WARPS;
    long long const u0 = warp * a.units / W;
    long long const u1 = (warp + 1) * a.units / W;
    int const n_units = static_cast<int>(u1 - u0);
    int const upt = a.ks / kUnit; // units per tile
    uint4 const* wp = a.p.w + u0 * 96 + lane;
    uint4* const xs4 = smem4;     // [upt][32][2] uint4
    uint4* const ring = smem4 + upt * 64 + wid * (S * 96) + lane;

#pragma unroll
    for (int u = 0; u < S; ++u)
    {
        if (u < n_units)
#pragma unroll
            for (int l = 0; l < 3; ++l)
                cp_async16(ring + (u * 3 + l) * 32, wp + (u * 3 + l) * 32);
        cp_async_commit();
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    {
        // x[n][8q..8q+7] -> word h = q & 1 of entries (s = q / 2, lane = n * 4 + j), entry index (s / 4, lane, s % 4).
        uint32_t* xs32 = reinterpret_cast<uint32_t*>(xs4);
        int const qn = a.ks * 2;
        int const items = kMaxRows * qn;
        constexpr int kSU = 8;
        for (int i0 = threadIdx.x; i0 < items; i0 += kSU * blockDim.x)
        {
            uint4 v[kSU];
#pragma unroll
            for (int u = 0; u < kSU; ++u)
            {
                int const i = i0 + u * blockDim.x;
                int const n = i / qn, q = i - n * qn;
                v[u] = (i < items && n < a.p.M) ? *reinterpret_cast<uint4 const*>(a.p.x + n * a.p.x_stride + q * 8)
                                                : make_uint4(0u, 0u, 0u, 0u);
            }
#pragma unroll
            for (int u = 0; u < kSU; ++u)
            {
                int const i = i0 + u * blockDim.x;
                if (i >= items)
                    continue;
                int const n = i / qn, q = i - n * qn;
                int const sstep = q >> 1;
                int const base = (((sstep >> 2) * 32 + n * 4) * 4 + (sstep & 3)) * 2 + (q & 1);
                xs32[base + 0] = v[u].x;
                xs32[base + 8] = v[u].y;
                xs32[base + 16] = v[u].z;
                xs32[base + 24] = v[u].w;
            }
        }
    }
    __syncthreads();
    asm volatile("griddepcontrol.launch_dependents;");

    uint4 const* xl = xs4 + lane * 2;
    int tile = static_cast<int>(u0 / upt);
    int ku = static_cast<int>(u0 % upt); // unit within the tile
    int done = 0, slot = 0;
    while (done < n_units)
    {
        int const seg = min(n_units - done, upt - ku);
        float ca[4] = {0.f, 0.f, 0.f, 0.f};
        float cb[4] = {0.f, 0.f, 0.f, 0.f};
        int const e0 = __ldg(a.p.exc_off + tile), e1 = __ldg(a.p.exc_off + tile + 1);
        for (int i = 0; i < seg; ++i)
        {
            cp_async_wait<S - 1>();
            uint4 const sm01 = ring[slot * 96];
            uint4 const sm23 = ring[slot * 96 + 32];
            uint4 const cw = ring[slot * 96 + 64];
            uint4 const xa = xl[(ku + i) * 64];
            uint4 const xb = xl[(ku + i) * 64 + 1];
            uint32_t const smw[8] = {sm01.x, sm01.y, sm01.z, sm01.w, sm23.x, sm23.y, sm23.z, sm23.w};
            uint32_t const cws[4] = {cw.x, cw.y, cw.z, cw.w};
            uint2 const xb2[4]
                = {make_uint2(xa.x, xa.y), make_uint2(xa.z, xa.w), make_uint2(xb.x, xb.y), make_uint2(xb.z, xb.w)};
#pragma unroll
            for (int ci = 0; ci < 4; ++ci)
            {
                uint4 wv;
                wv.x = decode_pair(smw[2 * ci], 0, cws[ci], 0, a.p.exp_base);
                wv.y = decode_pair(smw[2 * ci], 2, cws[ci], 1, a.p.exp_base);
                wv.z = decode_pair(smw[2 * ci + 1], 0, cws[ci], 2, a.p.exp_base);
                wv.w = decode_pair(smw[2 * ci + 1], 2, cws[ci], 3, a.p.exp_base);
                if (ci & 1)
                    mma_bf16(cb, wv, xb2[ci]);
                else
                    mma_bf16(ca, wv, xb2[ci]);
            }
            // Refill the slot only after its values were consumed (no ordering between ld.shared and cp.async).
            int const nxt = done + i + S;
            if (nxt < n_units)
#pragma unroll
                for (int l = 0; l < 3; ++l)
                    cp_async16(ring + (slot * 3 + l) * 32, wp + (nxt * 3 + l) * 32);
            cp_async_commit();
            slot = slot + 1 == S ? 0 : slot + 1;
        }
#pragma unroll
        for (int e = 0; e < 4; ++e)
            ca[e] += cb[e];
        // Exceptions of this tile with k in the segment: lanes load 32 at a time, each is broadcast; the owner lanes
        // (row g or g + 8) add the exact product x[n][k] * w with x from shared memory.
        {
            int const g = lane >> 2, j = lane & 3;
            int const kb = ku * 64, ke = (ku + seg) * 64;
            uint32_t const* xs32 = reinterpret_cast<uint32_t const*>(xs4);
            for (int b0 = e0; b0 < e1; b0 += 32)
            {
                uint32_t pos = 0xffffffffu;
                float2 wv = make_float2(0.f, 0.f);
                if (b0 + lane < e1)
                {
                    pos = __ldg(a.p.exc_pos + b0 + lane);
                    wv = __ldg(a.p.exc_w + b0 + lane);
                }
                int const cnt = min(32, e1 - b0);
                for (int q = 0; q < cnt; ++q)
                {
                    uint32_t const pq = __shfl_sync(0xffffffffu, pos, q);
                    float const w0 = __shfl_sync(0xffffffffu, wv.x, q);
                    int const r = static_cast<int>(pq >> 16), k = static_cast<int>(pq & 0xffffu);
                    // No early continue: every lane reaches each iteration's shuffles together.
                    if (k >= kb && k < ke && (r & 7) == g)
                    {
                        int const st = k >> 4, rr = k & 15;
                        bool const hi = r >= 8;
#pragma unroll
                        for (int q2 = 0; q2 < 2; ++q2)
                        {
                            int const n = 2 * j + q2;
                            uint32_t const word
                                = xs32[((((st >> 2) * 32 + n * 4 + ((rr & 7) >> 1)) * 4 + (st & 3)) * 2) + (rr >> 3)];
                            float const xv = __uint_as_float((rr & 1) ? (word & 0xffff0000u) : (word << 16));
                            float const d = __fmul_rn(xv, w0);
                            ca[q2] += hi ? 0.f : d;
                            ca[2 + q2] += hi ? d : 0.f;
                        }
                    }
                    __syncwarp();
                }
            }
        }
        flush(a, warp, tile, ku * kUnit, (ku + seg) * kUnit, lane, ca);
        done += seg;
        ++tile;
        ku = 0;
    }
#endif
}

} // namespace

size_t targetHeadSmemBytes(int K)
{
    return static_cast<size_t>(K / 64) * 64 * 16 + static_cast<size_t>(kWarps) * kStages * 96 * 16;
}

cudaError_t invokeTargetHead(Params const& p, int K, int grid, bool pdl, cudaStream_t stream)
{
    if (p.M < 1 || p.M > kMaxRows || K % 64 != 0 || p.N < 1 || grid < 1)
        return cudaErrorInvalidValue;
    Args a;
    a.p = p;
    a.tiles = (p.N + 15) / 16;
    a.ks = K / 16;
    a.units = static_cast<long long>(a.tiles) * a.ks / kUnit;
    // Every warp must own at least one unit: a split tile's finisher counts the warps between its first and last
    // owner, so a warp with an empty range in between would leave the tile unstored.
    if (a.units < kWarps)
        return cudaErrorInvalidValue;
    grid = static_cast<int>(std::min<long long>(grid, a.units / kWarps));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(grid));
    cfg.blockDim = dim3(kWarps * 32);
    cfg.dynamicSmemBytes = targetHeadSmemBytes(K);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    auto kern = coded_kernel<kWarps, kStages>;
    cudaError_t const err = cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(cfg.dynamicSmemBytes));
    if (err != cudaSuccess)
        return err;
    return cudaLaunchKernelEx(&cfg, kern, a);
}

} // namespace targetHead
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
