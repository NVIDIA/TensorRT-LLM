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

// TAILGLUE_A (see tailGlueSm107.h for the contract): one launch of 128-thread CTAs of three kinds, by blockIdx.x:
//   [0, storeBlocks)                      the context store, four warps per CTA, one warp per (row, layer, kv head);
//   [storeBlocks, + advBlocks)            the advance + forward indices, one thread per request;
//   [.., + n * nch)                       the read view, one CTA per (request, 512-column chunk), 4 columns / thread.
// The store's arithmetic is invokeDsparkCtxStore's (mul.rn / fma.rn, bf16 round-to-nearest-even), the served shape
// (head dim 128, rotary half a multiple of 4, kp1 4 / 8, page 32 / 64, aligned operands) with 4 contiguous elements per
// lane, every other shape one element per lane per step. Integer lines are the advance / read-view / indices kernels'.

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/csrc/tailGlueSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_bf16.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kViewItems = 4;
constexpr int kViewCols = kThreads * kViewItems; // 512 columns per read-view CTA
constexpr int kServedHeadDim = 128;
constexpr int kServedEpl = kServedHeadDim / 32;  // elements per lane

__device__ __forceinline__ float mulRn(float a, float b)
{
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fmaRn(float a, float b, float c)
{
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

__device__ __forceinline__ int64_t min64(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

__device__ __forceinline__ int64_t max64(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

__device__ __forceinline__ float wid(uint16_t x)
{
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
}

__device__ __forceinline__ uint16_t nar(float v)
{
    return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

__device__ __forceinline__ int64_t ldIdx(void const* p, bool is64, int64_t i)
{
    return is64 ? static_cast<int64_t const*>(p)[i] : static_cast<int64_t>(static_cast<int32_t const*>(p)[i]);
}

__device__ __forceinline__ void pdlEntry()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

struct Bf4
{
    uint16_t e[4];
};

__device__ __forceinline__ Bf4 ld4(uint16_t const* p)
{
    uint2 const w = *reinterpret_cast<uint2 const*>(p);
    return Bf4{{static_cast<uint16_t>(w.x), static_cast<uint16_t>(w.x >> 16), static_cast<uint16_t>(w.y),
        static_cast<uint16_t>(w.y >> 16)}};
}

__device__ __forceinline__ void st4(uint16_t* p, Bf4 const& v)
{
    uint2 w;
    w.x = v.e[0] | (static_cast<uint32_t>(v.e[1]) << 16);
    w.y = v.e[2] | (static_cast<uint32_t>(v.e[3]) << 16);
    *reinterpret_cast<uint2*>(p) = w;
}

__device__ __forceinline__ uint16_t* layerCache(TailGlueParams const& a, int layer)
{
    return a.ptrList ? reinterpret_cast<uint16_t*>(static_cast<int64_t const*>(a.cache)[layer])
                     : static_cast<uint16_t*>(a.cache) + static_cast<int64_t>(layer) * a.cacheStrideLayer;
}

// The served store program (r, layer, h) on one warp: 4 contiguous elements per lane.
template <int KP1, int PS, bool FOLD>
__device__ __forceinline__ void storeServed(TailGlueParams const& a, int prog, int lane)
{
    constexpr int HD = kServedHeadDim;
    constexpr int EPL = kServedEpl;
    int const nkv = a.numKvHeads;
    int const h = prog % nkv;
    int const layer = (prog / nkv) % a.numLayers;
    int const r32 = prog / (nkv * a.numLayers);
    int64_t const r = r32;
    int const rot = a.rot;
    int const rhalf = rot / 2;
    int const d0 = lane * EPL;
    bool const roped = d0 < rot;
    int const p0 = d0 < rhalf ? d0 + rhalf : d0 - rhalf;
    int const i0 = d0 < rhalf ? d0 : d0 - rhalf;
    uint16_t* cache = layerCache(a, layer);

    int const req32 = r32 / KP1;
    int64_t const req = req32;
    int const off = r32 - req32 * KP1;
    bool const lastRow = off == KP1 - 1;
    int64_t const p = a.pos[r];
    int64_t const pNxt = a.pos[lastRow ? r : r + 1];
    int64_t const nacc = ldIdx(a.nacc, a.naccIs64, req);
    int64_t const row = a.rows[req];
    uint16_t const* kbase = static_cast<uint16_t const*>(a.k) + ((r * a.numLayers + layer) * nkv + h) * HD;
    uint16_t const* vbase = static_cast<uint16_t const*>(a.kv) + r * a.kvStrideRow
        + (static_cast<int64_t>(layer) * 2 * nkv + nkv + h) * HD;
    Bf4 wSelf{}, wPartner{};
    if constexpr (FOLD)
    {
        uint16_t const* wbase = static_cast<uint16_t const*>(a.w) + static_cast<int64_t>(layer) * HD;
        wSelf = ld4(wbase + d0);
        if (roped)
            wPartner = ld4(wbase + p0);
    }
    Bf4 const kSelf = ld4(kbase + d0);
    Bf4 const vSelf = ld4(vbase + d0);
    Bf4 kPartner = kSelf;
    if (roped)
        kPartner = ld4(kbase + p0);

    float cv[EPL] = {}, sv[EPL] = {};
    if (roped)
    {
        float const* csr = a.cosSin + p * rot;
        float4 const c = *reinterpret_cast<float4 const*>(csr + i0);
        float4 const s = *reinterpret_cast<float4 const*>(csr + rhalf + i0);
        cv[0] = c.x;
        cv[1] = c.y;
        cv[2] = c.z;
        cv[3] = c.w;
        sv[0] = s.x;
        sv[1] = s.y;
        sv[2] = s.z;
        sv[3] = s.w;
    }
    int64_t const cap = a.counts[row] * PS - 1;
    int64_t const col = max64(min64(p, cap), 0);
    int64_t const nxt = max64(min64(pNxt, cap), 0);
    if (!(lastRow || nxt != col))
    {
        return; // a later row of this request writes the shared column
    }
    int64_t const page = a.table[row * a.tableStride + col / PS];

    float const m = off < nacc ? 1.0f : 0.0f;
    Bf4 kout, vout;
#pragma unroll
    for (int e = 0; e < EPL; ++e)
    {
        uint16_t x = kSelf.e[e];
        if constexpr (FOLD)
            x = nar(mulRn(wid(x), wid(wSelf.e[e])));
        if (roped)
        {
            uint16_t y = kPartner.e[e];
            if constexpr (FOLD)
                y = nar(mulRn(wid(y), wid(wPartner.e[e])));
            bool const lo = d0 < rhalf;
            float const av = lo ? wid(x) : wid(y);
            float const bv = lo ? wid(y) : wid(x);
            float const rt = lo ? fmaRn(av, cv[e], mulRn(-bv, sv[e])) : fmaRn(bv, cv[e], mulRn(av, sv[e]));
            kout.e[e] = nar(mulRn(wid(nar(rt)), m));
        }
        else
        {
            kout.e[e] = nar(mulRn(wid(x), m));
        }
        vout.e[e] = nar(mulRn(wid(vSelf.e[e]), m));
    }
    int64_t const slot = col % PS;
    st4(cache + ((page * 2 * nkv + h) * PS + slot) * HD + d0, kout);
    st4(cache + ((page * 2 * nkv + nkv + h) * PS + slot) * HD + d0, vout);
}

// The generic store program (r, layer, h) on one warp: any supported shape, one element per lane per step.
__device__ __forceinline__ void storeGeneric(TailGlueParams const& a, int prog, int lane)
{
    int const HD = a.headDim;
    int const nkv = a.numKvHeads;
    int const h = prog % nkv;
    int const layer = (prog / nkv) % a.numLayers;
    int64_t const r = prog / (nkv * a.numLayers);
    int const rot = a.rot;
    int const rhalf = rot / 2;
    int const ps = a.storePage;
    if (lane >= HD)
    {
        return;
    }
    uint16_t* cache = layerCache(a, layer);
    int64_t const req = r / a.kp1;
    int64_t const off = r - req * a.kp1;
    bool const lastRow = off == a.kp1 - 1;
    int64_t const p = a.pos[r];
    int64_t const pNxt = a.pos[lastRow ? r : r + 1];
    int64_t const nacc = ldIdx(a.nacc, a.naccIs64, req);
    int64_t const row = a.rows[req];
    int64_t const cap = a.counts[row] * ps - 1;
    int64_t const col = max64(min64(p, cap), 0);
    int64_t const nxt = max64(min64(pNxt, cap), 0);
    if (!(lastRow || nxt != col))
    {
        return;
    }
    int64_t const page = a.table[row * a.tableStride + col / ps];
    int64_t const slot = col % ps;
    float const m = off < nacc ? 1.0f : 0.0f;
    auto const* kbase = static_cast<uint16_t const*>(a.k) + ((r * a.numLayers + layer) * nkv + h) * HD;
    auto const* vbase = static_cast<uint16_t const*>(a.kv) + r * a.kvStrideRow
        + (static_cast<int64_t>(layer) * 2 * nkv + nkv + h) * HD;
    auto const* wbase = static_cast<uint16_t const*>(a.w) + static_cast<int64_t>(layer) * HD;
    float const* csr = a.cosSin + p * rot;
    uint16_t* kdst = cache + ((page * 2 * nkv + h) * ps + slot) * HD;
    uint16_t* vdst = cache + ((page * 2 * nkv + nkv + h) * ps + slot) * HD;
    for (int d = lane; d < HD; d += 32)
    {
        uint16_t x = kbase[d];
        if (a.foldW)
            x = nar(mulRn(wid(x), wid(wbase[d])));
        uint16_t ko;
        if (d < rot)
        {
            bool const lo = d < rhalf;
            int const q = lo ? d + rhalf : d - rhalf;
            int const i = lo ? d : d - rhalf;
            uint16_t y = kbase[q];
            if (a.foldW)
                y = nar(mulRn(wid(y), wid(wbase[q])));
            float const c = csr[i];
            float const s = csr[rhalf + i];
            float const av = lo ? wid(x) : wid(y);
            float const bv = lo ? wid(y) : wid(x);
            float const rt = lo ? fmaRn(av, c, mulRn(-bv, s)) : fmaRn(bv, c, mulRn(av, s));
            ko = nar(mulRn(wid(nar(rt)), m));
        }
        else
        {
            ko = nar(mulRn(wid(x), m));
        }
        kdst[d] = ko;
        vdst[d] = nar(mulRn(wid(vbase[d]), m));
    }
}

// The advance of request r (the advance-only kernel's lines), then its forward indices (the indices kernel's lines
// for row r, numCtx = the advertised length just stored, batchIdx = rows, pagesPerSlot = width).
__device__ __forceinline__ void advanceAndIndices(TailGlueParams const& a, int32_t r32)
{
    int64_t const r = r32;
    int64_t const slot = a.slots[r];
    int64_t const row = a.rows[r];
    bool const isDummy = slot == a.dummySlot;
    int64_t newLen = min64(a.ctxLen[slot] + ldIdx(a.nacc, a.naccIs64, r), a.maxCtx);
    if (!isDummy)
    {
        a.ctxLen[slot] = newLen;
    }
    if (r == 0)
    {
        a.ctxLen[a.dummySlot] = 0;
    }
    newLen = isDummy ? 0 : newLen;
    int64_t const allocated = max64(a.counts[row] * a.page - a.blockSize, 0);
    int64_t const numCtx = min64(newLen, allocated);
    int64_t const base = a.basePages[slot];
    int64_t const outLen = max64(numCtx - base * a.page, 0);
    a.out[r] = outLen;

    int32_t const pagesPerSlot = static_cast<int32_t>(a.width);
    int32_t const s = static_cast<int32_t>(outLen);
    a.seqlens[r] = s;
    a.batchOut[r] = static_cast<int32_t>(row);
    int32_t const aft = s + a.iblock;
    a.after[r] = aft;
    int32_t lp = (aft - 1) % a.ipage;
    lp = lp < 0 ? lp + a.ipage : lp; // torch's floored modulo for a positive divisor
    a.last[r] = lp + 1;
    a.indptr[r32] = r32 * pagesPerSlot;
    if (r32 == a.n - 1)
    {
        a.indptr[a.n] = a.n * pagesPerSlot;
    }
    int32_t* app = a.append + static_cast<int64_t>(r32) * a.iblock;
    int32_t* qf = a.qflat + static_cast<int64_t>(r32) * a.iblock;
    for (int32_t j = 0; j < a.iblock; ++j)
    {
        app[j] = s + j;
        qf[j] = static_cast<int32_t>(ldIdx(a.qpos, a.qposIs64, r * a.q0 + j * a.q1));
    }
}

// The read view of (request r, chunk): the advance's rebase recomputed from pos[r * kp1] (= ctx_len[slot] before
// the advance), then the copy kernel's lines.
__device__ __forceinline__ void readView(TailGlueParams const& a, int64_t r, int64_t chunk)
{
    int64_t const slot = a.slots[r];
    int64_t const row = a.rows[r];
    bool const isDummy = slot == a.dummySlot;
    int64_t const old = a.pos[r * a.kp1];
    int64_t newLen = min64(old + ldIdx(a.nacc, a.naccIs64, r), a.maxCtx);
    newLen = isDummy ? 0 : newLen;
    int64_t const allocated = max64(a.counts[row] * a.page - a.blockSize, 0);
    int64_t const numCtx = min64(newLen, allocated);
    int64_t const rebase = min64(a.basePages[slot], numCtx / a.page);
    int64_t const c0 = chunk * kViewCols;
    int64_t const c1 = min64(c0 + kViewCols, a.width);
    int64_t const lastCol = a.width - 1;
    int32_t const* src = a.table + row * a.tableStride;
    int32_t* dst = a.readView + row * a.readStride;
    int32_t v[kViewItems];
#pragma unroll
    for (int i = 0; i < kViewItems; ++i)
    {
        int64_t const j = c0 + threadIdx.x + i * kThreads;
        if (j < c1)
        {
            v[i] = src[min64(j + rebase, lastCol)];
        }
    }
#pragma unroll
    for (int i = 0; i < kViewItems; ++i)
    {
        int64_t const j = c0 + threadIdx.x + i * kThreads;
        if (j < c1)
        {
            dst[j] = v[i];
        }
    }
}

// KP1 == 0: the generic store (every value at run time); otherwise the served store at KP1 / PS / FOLD.
template <int KP1, int PS, bool FOLD>
__global__ void __launch_bounds__(kThreads)
    tailGlueKernel(TailGlueParams const a, int32_t storeBlocks, int32_t advBlocks, int32_t nch)
{
    pdlEntry();
    int32_t const b = blockIdx.x;
    if (b < storeBlocks)
    {
        int const warp = threadIdx.x / 32;
        int const lane = threadIdx.x % 32;
        int const prog = b * kWarps + warp;
        int const numProg = a.n * a.kp1 * a.numLayers * a.numKvHeads;
        if (prog >= numProg)
        {
            return;
        }
        if constexpr (KP1 > 0)
        {
            // A pointer-list cache's layer addresses are known only here: a layer that is not 8-byte aligned (an
            // offset bf16 view) takes the generic store, which has the same arithmetic.
            if (a.ptrList
                && (reinterpret_cast<uintptr_t>(layerCache(a, (prog / a.numKvHeads) % a.numLayers)) & 7u) != 0)
            {
                storeGeneric(a, prog, lane);
            }
            else
            {
                storeServed<KP1, PS, FOLD>(a, prog, lane);
            }
        }
        else
        {
            storeGeneric(a, prog, lane);
        }
        return;
    }
    if (b < storeBlocks + advBlocks)
    {
        int32_t const r = (b - storeBlocks) * kThreads + threadIdx.x;
        if (r < a.n)
        {
            advanceAndIndices(a, r);
        }
        return;
    }
    int32_t const v = b - storeBlocks - advBlocks;
    readView(a, v / nch, v % nch);
}

bool aligned(void const* ptr, uintptr_t bytes)
{
    return (reinterpret_cast<uintptr_t>(ptr) & (bytes - 1)) == 0;
}

bool served(TailGlueParams const& p)
{
    return p.headDim == kServedHeadDim && (p.kp1 == 4 || p.kp1 == 8) && (p.storePage == 32 || p.storePage == 64)
        && (p.rot / 2) % kServedEpl == 0 && aligned(p.kv, 8) && aligned(p.k, 8) && (!p.foldW || aligned(p.w, 8))
        && aligned(p.cosSin, 16) && p.kvStrideRow % kServedEpl == 0
        && (p.ptrList || (aligned(p.cache, 8) && p.cacheStrideLayer % kServedEpl == 0));
}

template <int KP1, int PS>
auto pickFold(bool fold)
{
    return fold ? tailGlueKernel<KP1, PS, true> : tailGlueKernel<KP1, PS, false>;
}

} // namespace

cudaError_t invokeTailGlue(TailGlueParams const& p, bool enablePdl, cudaStream_t stream)
{
    if (p.n <= 0)
    {
        return cudaSuccess;
    }
    int64_t const numProg = static_cast<int64_t>(p.n) * p.kp1 * p.numLayers * p.numKvHeads;
    int64_t const storeBlocks = (numProg + kWarps - 1) / kWarps;
    int64_t const advBlocks = (p.n + kThreads - 1) / kThreads;
    int64_t const nch = (p.width + kViewCols - 1) / kViewCols;
    int64_t const total = storeBlocks + advBlocks + static_cast<int64_t>(p.n) * nch;
    if (p.width < 1 || total > 0x7fffffff)
    {
        return cudaErrorInvalidValue;
    }
    void (*kernel)(TailGlueParams, int32_t, int32_t, int32_t) = tailGlueKernel<0, 0, false>;
    if (served(p))
    {
        if (p.kp1 == 8)
        {
            kernel = p.storePage == 64 ? pickFold<8, 64>(p.foldW) : pickFold<8, 32>(p.foldW);
        }
        else
        {
            kernel = p.storePage == 64 ? pickFold<4, 64>(p.foldW) : pickFold<4, 32>(p.foldW);
        }
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(total));
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(
        &cfg, kernel, p, static_cast<int32_t>(storeBlocks), static_cast<int32_t>(advBlocks), static_cast<int32_t>(nch));
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
