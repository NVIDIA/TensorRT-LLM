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

// DSPARK_GLUE_CPP: C++ / CUDA replacements of two DSpark drafter Triton kernels (see dsparkGlueSm107.h for the exact
// contracts). The kernel bodies are the ones gated in SpecTail's DGLUE oracle / MB (bitwise vs the
// Triton kernels on every output; _advance 15.9 -> 7.1 us at the in-model block-table width).
// ADV_PAR (advance-only + read-view copy; median -0.93 us vs the Triton pair)
// and the drafter block's q / k RMSNorm + RoPE (+ paged K / V append) (VAR 1 / pre 0 / PDL / 4 warps),
// both bitwise vs the Triton kernels they replace on R1; the Triton kernels are deleted.

#include "minimax_m3/csrc/m31/dsparkGlueSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

constexpr int kAdvThreads = 1024;
constexpr int kAdvUnroll = 8;
constexpr int kIdxThreads = 256;

template <typename NaccT>
__global__ void __launch_bounds__(kAdvThreads) dsparkAdvanceKernel(int64_t const* __restrict__ slots,
    int64_t const* __restrict__ rows, NaccT const* __restrict__ nacc, int64_t* __restrict__ ctxLen,
    int64_t const* __restrict__ counts, int64_t const* __restrict__ basePages, int32_t const* __restrict__ tables,
    int32_t* __restrict__ readView, int64_t* __restrict__ out, int64_t width, int64_t tablesStride, int64_t readStride,
    int64_t maxCtx, int64_t dummySlot, int64_t page, int64_t blockSize)
{
    __shared__ int64_t sRebase;
    __shared__ int64_t sRow;
    int64_t const r = blockIdx.x;
    if (threadIdx.x == 0)
    {
        int64_t const slot = slots[r];
        int64_t const row = rows[r];
        bool const isDummy = slot == dummySlot;
        int64_t newLen = min(ctxLen[slot] + static_cast<int64_t>(nacc[r]), maxCtx);
        if (!isDummy)
        {
            ctxLen[slot] = newLen;
        }
        if (r == 0)
        {
            ctxLen[dummySlot] = 0;
        }
        newLen = isDummy ? 0 : newLen;
        int64_t const allocated = max(counts[row] * page - blockSize, int64_t{0});
        int64_t const numCtx = min(newLen, allocated);
        int64_t const base = basePages[slot];
        sRebase = min(base, numCtx / page);
        out[r] = max(numCtx - base * page, int64_t{0});
        sRow = row;
    }
    __syncthreads();
    int64_t const rebase = sRebase;
    int32_t const* src = tables + sRow * tablesStride;
    int32_t* dst = readView + sRow * readStride;
    int64_t const last = width - 1;
    int64_t j = threadIdx.x;
    for (; j + (kAdvUnroll - 1) * kAdvThreads < width; j += kAdvUnroll * kAdvThreads)
    {
        int32_t v[kAdvUnroll];
#pragma unroll
        for (int u = 0; u < kAdvUnroll; ++u)
        {
            v[u] = __ldg(src + min(j + u * kAdvThreads + rebase, last));
        }
#pragma unroll
        for (int u = 0; u < kAdvUnroll; ++u)
        {
            dst[j + u * kAdvThreads] = v[u];
        }
    }
    for (; j < width; j += kAdvThreads)
    {
        dst[j] = __ldg(src + min(j + rebase, last));
    }
}

template <typename CtxT, typename BidT, typename QT>
__global__ void __launch_bounds__(kIdxThreads) dsparkIndicesKernel(CtxT const* __restrict__ numCtx,
    BidT const* __restrict__ batchIdx, QT const* __restrict__ qpos, int64_t q0, int64_t q1,
    int32_t* __restrict__ seqlens, int32_t* __restrict__ batchOut, int32_t* __restrict__ after,
    int32_t* __restrict__ indptr, int32_t* __restrict__ last, int32_t* __restrict__ append, int32_t* __restrict__ qflat,
    int32_t B, int32_t block, int32_t pagesPerSlot, int32_t pageSize)
{
    for (int32_t b = threadIdx.x; b <= B; b += blockDim.x)
    {
        indptr[b] = b * pagesPerSlot;
        if (b < B)
        {
            int32_t const s = static_cast<int32_t>(numCtx[b]);
            seqlens[b] = s;
            batchOut[b] = static_cast<int32_t>(batchIdx[b]);
            int32_t const a = s + block;
            after[b] = a;
            int32_t m = (a - 1) % pageSize;
            m = m < 0 ? m + pageSize : m; // torch's floored modulo for a positive divisor
            last[b] = m + 1;
        }
    }
    for (int32_t t = threadIdx.x; t < B * block; t += blockDim.x)
    {
        int32_t const rb = t / block;
        int32_t const j = t - rb * block;
        append[t] = static_cast<int32_t>(numCtx[rb]) + j;
        qflat[t] = static_cast<int32_t>(qpos[rb * q0 + j * q1]);
    }
}

template <typename CtxT, typename BidT>
void launchIndices(CtxT const* numCtx, BidT const* batchIdx, void const* qpos, bool qposIs64, int64_t q0, int64_t q1,
    int32_t* seqlens, int32_t* batchOut, int32_t* after, int32_t* indptr, int32_t* last, int32_t* append,
    int32_t* qflat, int32_t B, int32_t block, int32_t pagesPerSlot, int32_t pageSize, cudaStream_t stream)
{
    if (qposIs64)
    {
        dsparkIndicesKernel<CtxT, BidT, int64_t><<<1, kIdxThreads, 0, stream>>>(numCtx, batchIdx,
            static_cast<int64_t const*>(qpos), q0, q1, seqlens, batchOut, after, indptr, last, append, qflat, B, block,
            pagesPerSlot, pageSize);
    }
    else
    {
        dsparkIndicesKernel<CtxT, BidT, int32_t><<<1, kIdxThreads, 0, stream>>>(numCtx, batchIdx,
            static_cast<int32_t const*>(qpos), q0, q1, seqlens, batchOut, after, indptr, last, append, qflat, B, block,
            pagesPerSlot, pageSize);
    }
}

template <typename CtxT>
void launchIndicesBid(CtxT const* numCtx, void const* batchIdx, bool batchIdxIs64, void const* qpos, bool qposIs64,
    int64_t q0, int64_t q1, int32_t* seqlens, int32_t* batchOut, int32_t* after, int32_t* indptr, int32_t* last,
    int32_t* append, int32_t* qflat, int32_t B, int32_t block, int32_t pagesPerSlot, int32_t pageSize,
    cudaStream_t stream)
{
    if (batchIdxIs64)
    {
        launchIndices(numCtx, static_cast<int64_t const*>(batchIdx), qpos, qposIs64, q0, q1, seqlens, batchOut, after,
            indptr, last, append, qflat, B, block, pagesPerSlot, pageSize, stream);
    }
    else
    {
        launchIndices(numCtx, static_cast<int32_t const*>(batchIdx), qpos, qposIs64, q0, q1, seqlens, batchOut, after,
            indptr, last, append, qflat, B, block, pagesPerSlot, pageSize, stream);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// ADV_PAR: advance-only (one thread per request) + read-view copy (grid (n, column chunks)).
// ---------------------------------------------------------------------------------------------------------------------

constexpr int kParThreads = 128; // p1w4: 4 warps per CTA in both launches
constexpr int kParItems = 4;     // columns per thread in the copy launch
constexpr int kParCols = kParThreads * kParItems;

__device__ __forceinline__ void pdlWaitThenRelease()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Wait for the predecessor grid (its writes visible), then release the dependents (they wait themselves).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

cudaLaunchConfig_t pdlLaunchConfig(
    dim3 grid, dim3 block, bool enablePdl, cudaStream_t stream, cudaLaunchAttribute* attr)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cfg;
}

template <typename NaccT>
__global__ void __launch_bounds__(kParThreads) dsparkAdvanceOnlyKernel(int64_t const* __restrict__ slots,
    int64_t const* __restrict__ rows, NaccT const* __restrict__ nacc, int64_t* ctxLen,
    int64_t const* __restrict__ counts, int64_t const* __restrict__ basePages, int64_t* __restrict__ out,
    int64_t* __restrict__ rebase, int64_t n, int64_t maxCtx, int64_t dummySlot, int64_t page, int64_t blockSize)
{
    pdlWaitThenRelease();
    int64_t const r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= n)
    {
        return;
    }
    int64_t const slot = slots[r];
    int64_t const row = rows[r];
    bool const isDummy = slot == dummySlot;
    int64_t newLen = min(ctxLen[slot] + static_cast<int64_t>(nacc[r]), maxCtx);
    if (!isDummy)
    {
        ctxLen[slot] = newLen;
    }
    if (r == 0)
    {
        ctxLen[dummySlot] = 0;
    }
    newLen = isDummy ? 0 : newLen;
    int64_t const allocated = max(counts[row] * page - blockSize, int64_t{0});
    int64_t const numCtx = min(newLen, allocated);
    int64_t const base = basePages[slot];
    rebase[r] = min(base, numCtx / page);
    out[r] = max(numCtx - base * page, int64_t{0});
}

__global__ void __launch_bounds__(kParThreads) dsparkReadViewKernel(int64_t const* __restrict__ rows,
    int64_t const* __restrict__ rebase, int32_t const* __restrict__ tables, int32_t* __restrict__ readView,
    int64_t width, int64_t tablesStride, int64_t readStride)
{
    pdlWaitThenRelease();
    int64_t const r = blockIdx.x;
    int64_t const c0 = static_cast<int64_t>(blockIdx.y) * kParCols;
    int64_t const c1 = min(c0 + kParCols, width);
    int64_t const row = rows[r];
    int64_t const rb = rebase[r];
    int64_t const last = width - 1;
    int32_t const* src = tables + row * tablesStride;
    int32_t* dst = readView + row * readStride;
    int32_t v[kParItems];
#pragma unroll
    for (int i = 0; i < kParItems; ++i)
    {
        int64_t const j = c0 + threadIdx.x + i * kParThreads;
        if (j < c1)
        {
            v[i] = src[min(j + rb, last)];
        }
    }
#pragma unroll
    for (int i = 0; i < kParItems; ++i)
    {
        int64_t const j = c0 + threadIdx.x + i * kParThreads;
        if (j < c1)
        {
            dst[j] = v[i];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// q / k RMSNorm + partial NeoX RoPE (+ paged HND K / V append). Every float op is inline PTX (nothing contracted or
// reordered), so the bits are the Triton kernel's (and flashinfer's) independent of the compile flags.
// ---------------------------------------------------------------------------------------------------------------------

constexpr int kQknrHeadDim = 128;
constexpr int kQknrWarps = 4;

__device__ __forceinline__ float qknrAdd(float a, float b)
{
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float qknrMul(float a, float b)
{
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float qknrFma(float a, float b, float c)
{
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

__device__ __forceinline__ float qknrRsqrt(float a)
{
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ __forceinline__ float qknrF(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 qknrBf(float v)
{
    return __float2bfloat16_rn(v);
}

//! The per-lane arithmetic, shared by both load paths. e8: this lane's 8 contiguous elements for the sum of squares
//! (lanes 0..15); xe / we: the elements / weights at [d0, d0 + 4); xae / wae and xbe / wbe: the RoPE pair halves at
//! [i0, i0 + 4) and [half + i0, half + i0 + 4); cv / sv: cos / sin at the pair index. Writes the 4 outputs.
__device__ __forceinline__ void qknrLane(int lane, int d0, int half, bool roped, float eps, __nv_bfloat16 const* e8,
    __nv_bfloat16 const* xe, __nv_bfloat16 const* we, __nv_bfloat16 const* xae, __nv_bfloat16 const* xbe,
    __nv_bfloat16 const* wae, __nv_bfloat16 const* wbe, float const* cv, float const* sv, __nv_bfloat16* outv)
{
    // 16 partial sums (lanes 0..15), each of 8 contiguous squares, sequential from 0.
    float s = 0.f;
    if (lane < 16)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            float const x = qknrF(e8[j]);
            s = qknrAdd(s, qknrMul(x, x));
        }
    }
    // Pair tree: s8[k] = s[2k] + s[2k + 1] (lanes 2k), s4 (lanes 4k), s2 (lanes 8k), total (lane 0).
    float const s8 = qknrAdd(s, __shfl_down_sync(0xffffffffu, s, 1));
    float const s4 = qknrAdd(s8, __shfl_down_sync(0xffffffffu, s8, 2));
    float const s2 = qknrAdd(s4, __shfl_down_sync(0xffffffffu, s4, 4));
    float const total = __shfl_sync(0xffffffffu, qknrAdd(s2, __shfl_down_sync(0xffffffffu, s2, 8)), 0);
    float const rstd = qknrRsqrt(qknrAdd(qknrMul(total, 1.0f / kQknrHeadDim), eps));
#pragma unroll
    for (int e = 0; e < 4; ++e)
    {
        if (!roped)
        {
            outv[e] = qknrBf(qknrMul(qknrMul(qknrF(xe[e]), rstd), qknrF(we[e])));
        }
        else
        {
            float const x1 = qknrF(qknrBf(qknrMul(qknrMul(qknrF(xae[e]), rstd), qknrF(wae[e]))));
            float const x2 = qknrF(qknrBf(qknrMul(qknrMul(qknrF(xbe[e]), rstd), qknrF(wbe[e]))));
            outv[e] = d0 < half ? qknrBf(qknrFma(x1, cv[e], qknrMul(-x2, sv[e])))
                                : qknrBf(qknrFma(x2, cv[e], qknrMul(x1, sv[e])));
        }
    }
}

//! One warp per (row t, head h); lane l owns output elements [4 l, 4 l + 4). kVec: unconditional read-only vector
//! loads with lane-clamped in-bounds indices (VAR 1); else element loads (any alignment, same bits).
template <bool kVec>
__global__ void __launch_bounds__(32 * kQknrWarps) dsparkQkNormRopeKernel(__nv_bfloat16 const* __restrict__ qkv,
    int64_t qkvStride, int32_t const* __restrict__ pos, float const* __restrict__ cs, int64_t csStride,
    __nv_bfloat16 const* __restrict__ wq, __nv_bfloat16 const* __restrict__ wk, __nv_bfloat16* __restrict__ qOut,
    __nv_bfloat16* __restrict__ kOut, float epsQ, float epsK, __nv_bfloat16* __restrict__ cache,
    int64_t cachePageStride, void const* bidx, void const* apos, void const* kvi, void const* kvp, int32_t idx64,
    int64_t vCol, int32_t T, int32_t HQ, int32_t HKV, int32_t half, int32_t ps, bool store)
{
    int const lane = threadIdx.x & 31;
    int64_t const prog = static_cast<int64_t>(blockIdx.x) * kQknrWarps + (threadIdx.x >> 5);
    int const H = HQ + HKV;
    if (prog >= static_cast<int64_t>(T) * H)
    {
        return;
    }
    int64_t const t = prog / H;
    int const h = static_cast<int>(prog % H);
    bool const isQ = h < HQ;
    __nv_bfloat16 const* base = qkv + t * qkvStride + static_cast<int64_t>(h) * kQknrHeadDim;
    __nv_bfloat16 const* w = isQ ? wq : wk;
    float const eps = isQ ? epsQ : epsK;
    int const d0 = lane * 4;
    bool const roped = d0 < 2 * half;
    int const i0 = d0 < half ? d0 : d0 - half; // the RoPE pair index of d0 (pair groups are 4-aligned: half % 4 == 0)
    int const ic = roped ? i0 : 0;             // clamped: every lane loads in bounds, selects afterwards
    pdlWaitThenRelease();

    alignas(8) __nv_bfloat16 outv[4];
    if constexpr (kVec)
    {
        int const ls = lane & 15;
        uint4 sraw = __ldg(reinterpret_cast<uint4 const*>(base + ls * 8));
        uint2 const xs = __ldg(reinterpret_cast<uint2 const*>(base + d0));
        uint2 const xa = __ldg(reinterpret_cast<uint2 const*>(base + ic));
        uint2 const xb = __ldg(reinterpret_cast<uint2 const*>(base + half + ic));
        uint2 const ws = __ldg(reinterpret_cast<uint2 const*>(w + d0));
        uint2 const wa = __ldg(reinterpret_cast<uint2 const*>(w + ic));
        uint2 const wb = __ldg(reinterpret_cast<uint2 const*>(w + half + ic));
        float const* csr = cs + static_cast<int64_t>(__ldg(pos + t)) * csStride;
        float4 const cs4 = __ldg(reinterpret_cast<float4 const*>(csr + ic));
        float4 const sn4 = __ldg(reinterpret_cast<float4 const*>(csr + half + ic));
        if (lane >= 16)
        {
            sraw = make_uint4(0u, 0u, 0u, 0u);
        }
        float const cv[4] = {cs4.x, cs4.y, cs4.z, cs4.w};
        float const sv[4] = {sn4.x, sn4.y, sn4.z, sn4.w};
        qknrLane(lane, d0, half, roped, eps, reinterpret_cast<__nv_bfloat16 const*>(&sraw),
            reinterpret_cast<__nv_bfloat16 const*>(&xs), reinterpret_cast<__nv_bfloat16 const*>(&ws),
            reinterpret_cast<__nv_bfloat16 const*>(&xa), reinterpret_cast<__nv_bfloat16 const*>(&xb),
            reinterpret_cast<__nv_bfloat16 const*>(&wa), reinterpret_cast<__nv_bfloat16 const*>(&wb), cv, sv, outv);
    }
    else
    {
        __nv_bfloat16 e8[8], xe[4], we[4], xae[4], xbe[4], wae[4], wbe[4];
        float cv[4], sv[4];
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            e8[j] = lane < 16 ? base[lane * 8 + j] : __float2bfloat16_rn(0.f);
        }
        float const* csr = cs + static_cast<int64_t>(pos[t]) * csStride;
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            xe[e] = base[d0 + e];
            we[e] = w[d0 + e];
            xae[e] = base[ic + e];
            xbe[e] = base[half + ic + e];
            wae[e] = w[ic + e];
            wbe[e] = w[half + ic + e];
            cv[e] = csr[ic + e];
            sv[e] = csr[half + ic + e];
        }
        qknrLane(lane, d0, half, roped, eps, e8, xe, we, xae, xbe, wae, wbe, cv, sv, outv);
    }

    __nv_bfloat16* obase = isQ ? qOut + (t * HQ + h) * kQknrHeadDim : kOut + (t * HKV + (h - HQ)) * kQknrHeadDim;
    if constexpr (kVec)
    {
        *reinterpret_cast<uint2*>(obase + d0) = *reinterpret_cast<uint2 const*>(outv);
    }
    else
    {
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            obase[d0 + e] = outv[e];
        }
    }
    if (store && !isQ)
    {
        int64_t const hk = h - HQ;
        // idx64 bit j: the j-th index tensor (bidx, apos, kvi, kvp) is int64, else int32 (any mix).
        auto ld = [&](void const* ptr, int bit, int64_t k) -> int64_t
        {
            return ((idx64 >> bit) & 1) ? static_cast<int64_t const*>(ptr)[k]
                                        : static_cast<int64_t>(static_cast<int32_t const*>(ptr)[k]);
        };
        int64_t const b = ld(bidx, 0, t);
        int64_t const p = ld(apos, 1, t);
        int64_t const page = ld(kvi, 2, ld(kvp, 3, b) + p / ps);
        int64_t const slot = p % ps;
        __nv_bfloat16* kdst = cache + page * cachePageStride + (hk * ps + slot) * kQknrHeadDim;
        __nv_bfloat16* vdst = cache + page * cachePageStride + ((HKV + hk) * ps + slot) * kQknrHeadDim;
        __nv_bfloat16 const* vsrc = qkv + t * qkvStride + vCol + hk * kQknrHeadDim;
        if constexpr (kVec)
        {
            *reinterpret_cast<uint2*>(kdst + d0) = *reinterpret_cast<uint2 const*>(outv);
            *reinterpret_cast<uint2*>(vdst + d0) = *reinterpret_cast<uint2 const*>(vsrc + d0);
        }
        else
        {
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                kdst[d0 + e] = outv[e];
                vdst[d0 + e] = vsrc[d0 + e];
            }
        }
    }
}

} // namespace

void invokeDsparkAdvance(int64_t const* slots, int64_t const* rows, void const* nacc, bool naccIs64, int64_t* ctxLen,
    int64_t const* counts, int64_t const* basePages, int32_t const* tables, int32_t* readView, int64_t* out, int64_t n,
    int64_t width, int64_t tablesStride, int64_t readStride, int64_t maxCtx, int64_t dummySlot, int64_t page,
    int64_t blockSize, cudaStream_t stream)
{
    if (n <= 0)
    {
        return;
    }
    if (naccIs64)
    {
        dsparkAdvanceKernel<int64_t><<<static_cast<unsigned>(n), kAdvThreads, 0, stream>>>(slots, rows,
            static_cast<int64_t const*>(nacc), ctxLen, counts, basePages, tables, readView, out, width, tablesStride,
            readStride, maxCtx, dummySlot, page, blockSize);
    }
    else
    {
        dsparkAdvanceKernel<int32_t><<<static_cast<unsigned>(n), kAdvThreads, 0, stream>>>(slots, rows,
            static_cast<int32_t const*>(nacc), ctxLen, counts, basePages, tables, readView, out, width, tablesStride,
            readStride, maxCtx, dummySlot, page, blockSize);
    }
}

void invokeDsparkIndices(void const* numCtx, bool numCtxIs64, void const* batchIdx, bool batchIdxIs64, void const* qpos,
    bool qposIs64, int64_t q0, int64_t q1, int32_t* seqlens, int32_t* batchOut, int32_t* after, int32_t* indptr,
    int32_t* last, int32_t* append, int32_t* qflat, int32_t B, int32_t block, int32_t pagesPerSlot, int32_t pageSize,
    cudaStream_t stream)
{
    if (numCtxIs64)
    {
        launchIndicesBid(static_cast<int64_t const*>(numCtx), batchIdx, batchIdxIs64, qpos, qposIs64, q0, q1, seqlens,
            batchOut, after, indptr, last, append, qflat, B, block, pagesPerSlot, pageSize, stream);
    }
    else
    {
        launchIndicesBid(static_cast<int32_t const*>(numCtx), batchIdx, batchIdxIs64, qpos, qposIs64, q0, q1, seqlens,
            batchOut, after, indptr, last, append, qflat, B, block, pagesPerSlot, pageSize, stream);
    }
}

cudaError_t invokeDsparkAdvancePar(int64_t const* slots, int64_t const* rows, void const* nacc, bool naccIs64,
    int64_t* ctxLen, int64_t const* counts, int64_t const* basePages, int32_t const* tables, int32_t* readView,
    int64_t* out, int64_t* rebase, int64_t n, int64_t width, int64_t tablesStride, int64_t readStride, int64_t maxCtx,
    int64_t dummySlot, int64_t page, int64_t blockSize, bool enablePdl, cudaStream_t stream)
{
    if (n <= 0)
    {
        return cudaSuccess;
    }
    if (page < 1 || width < 0 || (width + kParCols - 1) / kParCols > 65535)
    {
        return cudaErrorInvalidValue;
    }
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = pdlLaunchConfig(
        dim3(static_cast<unsigned>((n + kParThreads - 1) / kParThreads)), dim3(kParThreads), enablePdl, stream, attr);
    cudaError_t err = naccIs64
        ? cudaLaunchKernelEx(&cfg, dsparkAdvanceOnlyKernel<int64_t>, slots, rows, static_cast<int64_t const*>(nacc),
            ctxLen, counts, basePages, out, rebase, n, maxCtx, dummySlot, page, blockSize)
        : cudaLaunchKernelEx(&cfg, dsparkAdvanceOnlyKernel<int32_t>, slots, rows, static_cast<int32_t const*>(nacc),
            ctxLen, counts, basePages, out, rebase, n, maxCtx, dummySlot, page, blockSize);
    if (err != cudaSuccess || width == 0)
    {
        return err; // width 0: the advance still runs, the copy is empty (as the Triton pair)
    }
    cfg = pdlLaunchConfig(dim3(static_cast<unsigned>(n), static_cast<unsigned>((width + kParCols - 1) / kParCols)),
        dim3(kParThreads), enablePdl, stream, attr);
    return cudaLaunchKernelEx(&cfg, dsparkReadViewKernel, static_cast<int64_t const*>(rows),
        static_cast<int64_t const*>(rebase), tables, readView, width, tablesStride, readStride);
}

cudaError_t invokeDsparkQkNormRope(__nv_bfloat16 const* qkv, int64_t qkvStride, int32_t const* pos, float const* cs,
    int64_t csStride, __nv_bfloat16 const* wq, __nv_bfloat16 const* wk, __nv_bfloat16* qOut, __nv_bfloat16* kOut,
    float epsQ, float epsK, __nv_bfloat16* cache, int64_t cachePageStride, void const* bidx, void const* apos,
    void const* kvi, void const* kvp, int32_t idx64, int64_t vCol, int32_t T, int32_t HQ, int32_t HKV, int32_t half,
    int32_t ps, bool store, bool enablePdl, cudaStream_t stream)
{
    if (T <= 0)
    {
        return cudaSuccess;
    }
    if (half < 4 || 2 * half > kQknrHeadDim || half % 4 != 0 || HQ < 0 || HKV < 1 || (store && ps < 1))
    {
        return cudaErrorInvalidValue; // the 4-element lane groups need 4-aligned RoPE pair groups
    }
    auto misaligned = [](void const* p, uintptr_t mask) { return (reinterpret_cast<uintptr_t>(p) & mask) != 0; };
    // The vector path's loads / stores: 16-B qkv rows (uint4), 16-B cos / sin rows (float4), 8-B weights, outputs,
    // V columns and cache rows (uint2). Anything else takes the element-load instantiation (same bits).
    bool const vec = !misaligned(qkv, 15) && (qkvStride & 7) == 0 && !misaligned(cs, 15) && (csStride & 3) == 0
        && (vCol & 3) == 0 && !misaligned(wq, 7) && !misaligned(wk, 7) && !misaligned(qOut, 7) && !misaligned(kOut, 7)
        && (!store || (!misaligned(cache, 7) && (cachePageStride & 3) == 0));
    int64_t const progs = static_cast<int64_t>(T) * (HQ + HKV);
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = pdlLaunchConfig(dim3(static_cast<unsigned>((progs + kQknrWarps - 1) / kQknrWarps)),
        dim3(32 * kQknrWarps), enablePdl, stream, attr);
    return vec
        ? cudaLaunchKernelEx(&cfg, dsparkQkNormRopeKernel<true>, qkv, qkvStride, pos, cs, csStride, wq, wk, qOut, kOut,
            epsQ, epsK, cache, cachePageStride, bidx, apos, kvi, kvp, idx64, vCol, T, HQ, HKV, half, ps, store)
        : cudaLaunchKernelEx(&cfg, dsparkQkNormRopeKernel<false>, qkv, qkvStride, pos, cs, csStride, wq, wk, qOut, kOut,
            epsQ, epsK, cache, cachePageStride, bidx, apos, kvi, kvp, idx64, vCol, T, HQ, HKV, half, ps, store);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
