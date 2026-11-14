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

// DSPARK_CTX_STORE_CPP (no Triton in served paths): CUDA C++ replacement of the Triton _ctx_store_kernel of
// ops/dspark/dspark_ctx_store.py (see dsparkCtxStore.h for the contract). Gated bitwise vs the Triton kernel
// and timed on R1 (EXACT on every cell; served shape p1w32 median -0.291 us / call,
// max -0.008 us vs Triton). The decomposition is the Triton launch's: grid (n, L, NKV), one 32-lane warp per (row,
// layer, kv head).
// - Served build (head_dim 128, columns from positions, the bound pool's per-layer list or the private arena's
//   stacked cache, KP1 4 / 8, page 32 / 64, int64 positions / rows / counts, int32 block table): 4 contiguous elements
//   per lane (8-byte K / V / RoPE-partner loads and stores, 16-byte cos / sin loads), KP1 / page size / modes compile
//   time (shifts and masks). The k_norm weight and the pointer-list layer base (constants no predecessor writes) load
//   before griddepcontrol.wait; after it every index / K / partner / V load issues at once; one dependent round trip
//   for cos / sin and the page id (+ the counts on the bound pool, where the page id is speculated at col = max(pos, 0)
//   and re-read only when the clamp moved the column).
// - Generic build (every other shape / mode / index dtype): one element per lane per step, everything at run time.
// The arithmetic is the Triton kernel's, op for op (mul.rn / fma.rn, bf16 round-to-nearest-even), in both builds.

#include "minimax_m3/csrc/m31/dsparkCtxStore.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

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

// bf16 bits <-> fp32: exact widen, round-to-nearest-even narrow
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

constexpr int kServedHeadDim = 128;
constexpr int kServedEpl = kServedHeadDim / 32; // elements per lane

// The served build (see the file comment). Host-guaranteed: headDim 128, colFromPos, (rot / 2) % 4 == 0, the K / V /
// weight / stacked cache 8-byte aligned with row strides multiples of 4 elements, cos / sin 16-byte aligned, int64
// positions / rows / counts, int32 block table (numAccepted int32 or int64).
template <int KP1, int PS, bool PTR, bool BOUND, bool FOLD>
__global__ void __launch_bounds__(32) dsparkCtxStoreServedKernel(DsparkCtxStoreParams const a)
{
    constexpr int HD = kServedHeadDim;
    constexpr int EPL = kServedEpl;
    int const rot = a.rot;
    int const rhalf = rot / 2;
    int const lane = threadIdx.x;
    int const h = blockIdx.z;
    int const layer = blockIdx.y;
    int const r32 = blockIdx.x;
    int64_t const r = r32;
    int const d0 = lane * EPL;
    bool const roped = d0 < rot;                         // a lane's 4 elements are all roped or all not
    int const p0 = d0 < rhalf ? d0 + rhalf : d0 - rhalf; // RoPE partner
    int const i0 = d0 < rhalf ? d0 : d0 - rhalf;         // cos / sin index
    uint16_t const* wbase = static_cast<uint16_t const*>(a.w) + static_cast<int64_t>(layer) * HD;

    // ---- before the wait: layer constants (k_norm weight own + partner; the pointer-list layer base)
    Bf4 wown{}, wpar{};
    if constexpr (FOLD)
    {
        wown = ld4(wbase + d0);
        if (roped)
            wpar = ld4(wbase + p0);
    }
    uint16_t* cache;
    if constexpr (PTR)
        cache = reinterpret_cast<uint16_t*>(static_cast<int64_t const*>(a.cache)[layer]);
    else
        cache = static_cast<uint16_t*>(a.cache) + static_cast<int64_t>(layer) * a.cacheStrideLayer;
    pdlEntry();

    // ---- round trip 1: every independent load
    int const req32 = r32 / KP1;
    int64_t const req = req32;
    int const off = r32 - req32 * KP1;
    bool const last = off == KP1 - 1;
    int64_t const rn = last ? r : r + 1;
    auto const* pos = static_cast<int64_t const*>(a.pos);
    int64_t const p = pos[r];
    int64_t const pNxt = pos[rn];
    int64_t const nacc = ldIdx(a.numAccepted, (a.idxBits >> 1) & 1, req);
    int64_t const row = static_cast<int64_t const*>(a.requestRows)[req];
    uint16_t const* kbase = static_cast<uint16_t const*>(a.k) + ((r * a.numLayers + layer) * a.numKvHeads + h) * HD;
    uint16_t const* vbase = static_cast<uint16_t const*>(a.kv) + r * a.kvStrideRow
        + (static_cast<int64_t>(layer) * 2 * a.numKvHeads + a.numKvHeads + h) * HD;
    Bf4 const kSelf = ld4(kbase + d0);
    Bf4 const vown = ld4(vbase + d0);
    Bf4 kPeer = kSelf;
    if (roped)
        kPeer = ld4(kbase + p0);

    // ---- round trip 2: cos / sin, the page id (+ the counts on the bound pool)
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
    auto const* table = static_cast<int32_t const*>(a.table);
    int64_t col, nxt, page;
    if constexpr (BOUND)
    {
        int64_t const pc = max64(p, 0);
        int64_t const pcPage = pc / PS;
        bool const spec = pcPage < a.tableWidth;
        int64_t const pageSpec = spec ? static_cast<int64_t>(table[row * a.tableStride + pcPage]) : 0;
        int64_t const cap = static_cast<int64_t const*>(a.counts)[row] * PS - 1;
        col = max64(min64(p, cap), 0);
        nxt = max64(min64(pNxt, cap), 0);
        page = (spec && col == pc) ? pageSpec : static_cast<int64_t>(table[row * a.tableStride + col / PS]);
    }
    else
    {
        col = min64(p, a.maxCol);
        nxt = min64(pNxt, a.maxCol);
        page = table[row * a.tableStride + col / PS];
    }
    if (!(last || nxt != col))
    {
        return; // a later row of this request writes the shared column
    }

    // ---- the math (the Triton kernel's op order)
    float const m = off < nacc ? 1.0f : 0.0f;
    Bf4 kout, vout;
#pragma unroll
    for (int e = 0; e < EPL; ++e)
    {
        uint16_t x = kSelf.e[e];
        if constexpr (FOLD)
            x = nar(mulRn(wid(x), wid(wown.e[e])));
        if (roped)
        {
            uint16_t y = kPeer.e[e];
            if constexpr (FOLD)
                y = nar(mulRn(wid(y), wid(wpar.e[e])));
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
        vout.e[e] = nar(mulRn(wid(vown.e[e]), m));
    }
    int64_t const slot = col % PS;
    st4(cache + ((page * 2 * a.numKvHeads + h) * PS + slot) * HD + d0, kout);
    st4(cache + ((page * 2 * a.numKvHeads + a.numKvHeads + h) * PS + slot) * HD + d0, vout);
}

// The generic build: any supported shape, mode and index dtypes, one element per lane per step.
__global__ void __launch_bounds__(32) dsparkCtxStoreGenericKernel(DsparkCtxStoreParams const a)
{
    int const HD = a.headDim;
    int const rot = a.rot;
    int const rhalf = rot / 2;
    int const lane = threadIdx.x;
    int const h = blockIdx.z;
    int const layer = blockIdx.y;
    int64_t const r = blockIdx.x;
    uint16_t* cache = a.ptrList ? reinterpret_cast<uint16_t*>(static_cast<int64_t const*>(a.cache)[layer])
                                : static_cast<uint16_t*>(a.cache) + static_cast<int64_t>(layer) * a.cacheStrideLayer;
    pdlEntry();
    if (lane >= HD)
    {
        return;
    }
    auto bit = [&](int which) { return ((a.idxBits >> which) & 1) != 0; };
    int64_t const req = r / a.kp1;
    int64_t const off = r - req * a.kp1;
    bool const last = off == a.kp1 - 1;
    int64_t const rn = last ? r : r + 1;
    int64_t const p = ldIdx(a.pos, bit(0), r);
    int64_t const nacc = ldIdx(a.numAccepted, bit(1), req);
    int64_t const row = ldIdx(a.requestRows, bit(2), req);
    int64_t col, nxt;
    if (a.colFromPos)
    {
        int64_t const pNxt = ldIdx(a.pos, bit(0), rn);
        if (a.bound)
        {
            int64_t const cap = ldIdx(a.counts, bit(4), row) * a.pageSize - 1;
            col = max64(min64(p, cap), 0);
            nxt = max64(min64(pNxt, cap), 0);
        }
        else
        {
            col = min64(p, a.maxCol);
            nxt = min64(pNxt, a.maxCol);
        }
    }
    else
    {
        col = ldIdx(a.cols, bit(3), r);
        nxt = ldIdx(a.cols, bit(3), rn);
    }
    if (!(last || nxt != col))
    {
        return;
    }
    int64_t const page = ldIdx(a.table, bit(5), row * a.tableStride + col / a.pageSize);
    int64_t const slot = col % a.pageSize;
    float const m = off < nacc ? 1.0f : 0.0f;
    auto const* kbase = static_cast<uint16_t const*>(a.k) + ((r * a.numLayers + layer) * a.numKvHeads + h) * HD;
    auto const* vbase = static_cast<uint16_t const*>(a.kv) + r * a.kvStrideRow
        + (static_cast<int64_t>(layer) * 2 * a.numKvHeads + a.numKvHeads + h) * HD;
    auto const* wbase = static_cast<uint16_t const*>(a.w) + static_cast<int64_t>(layer) * HD;
    float const* csr = a.cosSin + p * rot;
    uint16_t* kdst = cache + ((page * 2 * a.numKvHeads + h) * a.pageSize + slot) * HD;
    uint16_t* vdst = cache + ((page * 2 * a.numKvHeads + a.numKvHeads + h) * a.pageSize + slot) * HD;
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

template <typename Kernel>
void launchPdl(Kernel kernel, DsparkCtxStoreParams const& p, cudaStream_t stream)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim
        = dim3(static_cast<unsigned>(p.n), static_cast<unsigned>(p.numLayers), static_cast<unsigned>(p.numKvHeads));
    cfg.blockDim = dim3(32);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p));
}

template <int KP1, int PS>
void launchServedModes(DsparkCtxStoreParams const& p, cudaStream_t stream)
{
    // the two in-model pairings: the bound pool = per-layer list, the private arena = one stacked tensor
    if (p.bound)
    {
        p.foldW ? launchPdl(dsparkCtxStoreServedKernel<KP1, PS, true, true, true>, p, stream)
                : launchPdl(dsparkCtxStoreServedKernel<KP1, PS, true, true, false>, p, stream);
    }
    else
    {
        p.foldW ? launchPdl(dsparkCtxStoreServedKernel<KP1, PS, false, false, true>, p, stream)
                : launchPdl(dsparkCtxStoreServedKernel<KP1, PS, false, false, false>, p, stream);
    }
}

bool aligned(void const* ptr, uintptr_t bytes)
{
    return (reinterpret_cast<uintptr_t>(ptr) & (bytes - 1)) == 0;
}

// idxBits of the served build: positions (bit 0), rows (2) int64; counts (4) int64 on the bound pool; table (5) int32;
// numAccepted (1) either.
bool servedIndices(DsparkCtxStoreParams const& p)
{
    return (p.idxBits & 1) && (p.idxBits & 4) && !(p.idxBits & 32) && (!p.bound || (p.idxBits & 16));
}

bool served(DsparkCtxStoreParams const& p)
{
    return p.headDim == kServedHeadDim && p.colFromPos && p.bound == p.ptrList && (p.kp1 == 4 || p.kp1 == 8)
        && (p.pageSize == 32 || p.pageSize == 64) && (p.rot / 2) % kServedEpl == 0 && servedIndices(p)
        && aligned(p.kv, 8) && aligned(p.k, 8) && (!p.foldW || aligned(p.w, 8)) && aligned(p.cosSin, 16)
        && p.kvStrideRow % kServedEpl == 0
        && (p.ptrList || (aligned(p.cache, 8) && p.cacheStrideLayer % kServedEpl == 0));
}

} // namespace

void invokeDsparkCtxStore(DsparkCtxStoreParams const& p, cudaStream_t stream)
{
    if (p.n <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(p.kp1 >= 1 && p.numLayers >= 1 && p.numLayers <= 65535 && p.numKvHeads >= 1
            && p.numKvHeads <= 65535 && p.headDim >= 16 && p.headDim <= 256 && (p.headDim & (p.headDim - 1)) == 0
            && p.rot >= 2 && p.rot % 2 == 0 && p.rot <= p.headDim && p.pageSize >= 1,
        "dspark ctx_store: unsupported shape");
    if (served(p))
    {
        if (p.kp1 == 8)
        {
            p.pageSize == 64 ? launchServedModes<8, 64>(p, stream) : launchServedModes<8, 32>(p, stream);
        }
        else
        {
            p.pageSize == 64 ? launchServedModes<4, 64>(p, stream) : launchServedModes<4, 32>(p, stream);
        }
        return;
    }
    launchPdl(dsparkCtxStoreGenericKernel, p, stream);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
