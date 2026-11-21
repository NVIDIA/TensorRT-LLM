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

// The DSpark / DFlash drafter's block-table refresh (ops/dspark/dspark_block_tables): CUDA C++ replacement of
// its former Triton kernels _refresh_kernel / _refresh_kernel_2d (NO TRITON in served paths), bit for bit.
// Measured (R1): EXACT on every input class (int32 / int64 offsets, widths
// 1..8196, power-of-two / odd / > 2^31 divisors, padded tables); this layout: median -0.449 / max +0.050 us
// per call vs Triton over every timed cell, -0.92 to -1.13 us at the served width 8,196 (int32, num_seqs 1 / 2 / 4).
// Launch: PDL on, 512 threads per wide CTA (the p1w4 arm).
//   tables[r, j] = int32(max(v, 0) // divisor), v = src[r, j]; counts[r] = number of v >= 0 over the row (STORED).
// width <= 256: a warp per row at ENT = 1 / 2 / 4 / 8 entries per lane (the smallest that covers the row), 4 rows per
// CTA, ballot counts. 256 < width <= 512 (e16w2): two warps per row at 8 entries per lane each, 4 rows (256 threads)
// per CTA, the second warp's count through one smem int per row and one barrier. Wider: grid (rows, 1 + chunks);
// CTA y = 0 counts the row (16-B loads, block reduction, one store), CTAs y >= 1 decode 2,048 entries each (16-B load /
// store when every row is 16-B aligned with a unit column stride, decided on the host). No atomics, fences or
// global scratch: every output has exactly one writer.

#include "minimax_m3/csrc/m31/dsparkBlockTables.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <type_traits>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{
constexpr int kMaxWpc = 8;                   // 128 x wpc threads per CTA: <= 1024
constexpr int kCntUnroll = 4;                // count CTA: 16-B loads per thread per pass
constexpr int kTinyMaxEnt = 16;              // tiny path: entries per lane at most
constexpr int kTinyWidth = 32 * kTinyMaxEnt; // 512
constexpr int kTinyRows = 4;                 // tiny path: rows (warps) per CTA, every ENT
constexpr int kE16W2Rows = 4;                // e16w2 (256 < width <= 512): rows per CTA (2 warps each)

template <typename T>
__device__ __forceinline__ int32_t decode(T v, int shift, T divisor)
{
    T const c = v > 0 ? v : T(0);
    if (shift >= 0)
        return static_cast<int32_t>(c >> shift);
    using UT = typename std::conditional<sizeof(T) == 8, uint64_t, uint32_t>::type;
    return static_cast<int32_t>(static_cast<T>(static_cast<UT>(c) / static_cast<UT>(divisor)));
}

__device__ __forceinline__ int32_t warp_sum(int32_t v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// 4 consecutive entries from a 16-B aligned address
template <typename T>
__device__ __forceinline__ void load4(T const* p, T (&v)[4])
{
    if constexpr (sizeof(T) == 4)
    {
        int4 const x = *reinterpret_cast<int4 const*>(p);
        v[0] = x.x;
        v[1] = x.y;
        v[2] = x.z;
        v[3] = x.w;
    }
    else
    {
        longlong2 const a = reinterpret_cast<longlong2 const*>(p)[0];
        longlong2 const b = reinterpret_cast<longlong2 const*>(p)[1];
        v[0] = a.x;
        v[1] = a.y;
        v[2] = b.x;
        v[3] = b.y;
    }
}

// entries >= 0 in one 16-B vector (int64: the sign is the high word's)
template <typename T>
__device__ __forceinline__ int32_t count_vec(int4 x)
{
    if constexpr (sizeof(T) == 4)
        return (x.x >= 0) + (x.y >= 0) + (x.z >= 0) + (x.w >= 0);
    else
        return (x.y >= 0) + (x.w >= 0);
}

// width <= 32 x ENT (ENT <= 8): ENT entries per lane, one warp per row, kTinyRows rows per CTA
template <typename T, int ENT>
__global__ void __launch_bounds__(32 * kTinyRows) dspark_refresh_tiny_kernel(T const* __restrict__ src,
    int64_t src_stride_row, int64_t src_stride_col, int32_t* __restrict__ tables, int64_t tables_stride,
    int64_t* __restrict__ counts, int rows, int width, T divisor, int shift)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    int const lane = threadIdx.x & 31;
    int64_t const r = static_cast<int64_t>(blockIdx.x) * kTinyRows + (threadIdx.x >> 5);
    if (r >= rows) // warp-uniform
        return;
    T const* srow = src + r * src_stride_row;
    int32_t* trow = tables + r * tables_stride;
    T v[ENT];
#pragma unroll
    for (int i = 0; i < ENT; ++i)
    {
        int const o = i * 32 + lane;
        v[i] = o < width ? srow[static_cast<int64_t>(o) * src_stride_col] : T(-1);
    }
    int cnt = 0;
#pragma unroll
    for (int i = 0; i < ENT; ++i)
    {
        int const o = i * 32 + lane;
        cnt += __popc(__ballot_sync(0xffffffffu, v[i] >= 0));
        if (o < width)
            trow[o] = decode(v[i], shift, divisor);
    }
    if (lane == 0)
        counts[r] = static_cast<int64_t>(cnt);
}

// e16w2: 256 < width <= 512, two warps per row (warp h covers entries [256 h, 256 h + 256), 8 per lane), kE16W2Rows
// rows per CTA; warp 1's count reaches warp 0 through one smem int per row and one barrier (every warp reaches it).
template <typename T>
__global__ void __launch_bounds__(64 * kE16W2Rows) dspark_refresh_tiny_w2_kernel(T const* __restrict__ src,
    int64_t src_stride_row, int64_t src_stride_col, int32_t* __restrict__ tables, int64_t tables_stride,
    int64_t* __restrict__ counts, int rows, int width, T divisor, int shift)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    __shared__ int32_t half_cnt[kE16W2Rows];
    constexpr int kHalf = kTinyMaxEnt / 2; // 8 entries per lane per warp
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const lr = warp >> 1;    // row within the CTA
    int const h = warp & 1;      // half of the row
    int64_t const r = static_cast<int64_t>(blockIdx.x) * kE16W2Rows + lr;
    bool const valid = r < rows; // warp-uniform
    int cnt = 0;
    if (valid)
    {
        T const* srow = src + r * src_stride_row;
        int32_t* trow = tables + r * tables_stride;
        T v[kHalf];
#pragma unroll
        for (int i = 0; i < kHalf; ++i)
        {
            int const o = (h * kHalf + i) * 32 + lane;
            v[i] = o < width ? srow[static_cast<int64_t>(o) * src_stride_col] : T(-1);
        }
#pragma unroll
        for (int i = 0; i < kHalf; ++i)
        {
            int const o = (h * kHalf + i) * 32 + lane;
            cnt += __popc(__ballot_sync(0xffffffffu, v[i] >= 0));
            if (o < width)
                trow[o] = decode(v[i], shift, divisor);
        }
        if (h == 1 && lane == 0)
            half_cnt[lr] = cnt;
    }
    __syncthreads();
    if (valid && h == 0 && lane == 0)
        counts[r] = static_cast<int64_t>(cnt + half_cnt[lr]);
}

template <typename T>
__device__ __forceinline__ void count_row(T const* srow, int64_t sc, int width, int64_t* out)
{
    __shared__ int32_t warp_cnt[32];
    int const tid = threadIdx.x;
    int const nthr = blockDim.x;
    int32_t cnt = 0;
    if (sc == 1)
    {
        constexpr int kE = 16 / static_cast<int>(sizeof(T));
        int head = static_cast<int>(
            ((16u - (static_cast<unsigned>(reinterpret_cast<uintptr_t>(srow)) & 15u)) & 15u) / sizeof(T));
        head = head < width ? head : width;
        if (tid < head)
            cnt += srow[tid] >= 0 ? 1 : 0;
        int const nvec = (width - head) / kE;
        int4 const* vb = reinterpret_cast<int4 const*>(srow + head);
        for (int base = 0; base < nvec; base += kCntUnroll * nthr)
        {
            int4 x[kCntUnroll];
#pragma unroll
            for (int u = 0; u < kCntUnroll; ++u)
            {
                int const i = base + u * nthr + tid;
                x[u] = i < nvec ? vb[i] : make_int4(-1, -1, -1, -1);
            }
#pragma unroll
            for (int u = 0; u < kCntUnroll; ++u)
                cnt += count_vec<T>(x[u]);
        }
        int const t0 = head + nvec * kE;
        if (tid < width - t0)
            cnt += srow[t0 + tid] >= 0 ? 1 : 0;
    }
    else
    {
        for (int o = tid; o < width; o += nthr)
            cnt += srow[static_cast<int64_t>(o) * sc] >= 0 ? 1 : 0;
    }
    cnt = warp_sum(cnt);
    if ((tid & 31) == 0)
        warp_cnt[tid >> 5] = cnt;
    __syncthreads();
    if (tid < 32)
    {
        int32_t s = tid < (nthr >> 5) ? warp_cnt[tid] : 0;
        s = warp_sum(s);
        if (tid == 0)
            *out = static_cast<int64_t>(s);
    }
}

template <typename T>
__device__ __forceinline__ void decode_chunk(
    T const* srow, int64_t sc, int32_t* trow, int width, int c, T divisor, int shift, bool vec)
{
    int const tid = threadIdx.x;
    int const nthr = blockDim.x;
    if (vec) // unit column stride, 16-B aligned src / tables rows
    {
        int const o = 4 * (c * nthr + tid);
        if (o + 3 < width)
        {
            T v[4];
            load4(srow + o, v);
            *reinterpret_cast<int4*>(trow + o) = make_int4(decode(v[0], shift, divisor), decode(v[1], shift, divisor),
                decode(v[2], shift, divisor), decode(v[3], shift, divisor));
        }
        else
        {
#pragma unroll
            for (int k = 0; k < 4; ++k)
                if (o + k < width)
                    trow[o + k] = decode(srow[o + k], shift, divisor);
        }
        return;
    }
    int const c0 = c * 4 * nthr;
    T v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        int const o = c0 + k * nthr + tid;
        v[k] = o < width ? srow[static_cast<int64_t>(o) * sc] : T(0);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        int const o = c0 + k * nthr + tid;
        if (o < width)
            trow[o] = decode(v[k], shift, divisor);
    }
}

// wider rows: blockIdx.y == 0 counts the row, blockIdx.y >= 1 decodes chunk blockIdx.y - 1 (4 x blockDim entries)
template <typename T>
__global__ void __launch_bounds__(128 * kMaxWpc) dspark_refresh_wide_kernel(T const* __restrict__ src,
    int64_t src_stride_row, int64_t src_stride_col, int32_t* __restrict__ tables, int64_t tables_stride,
    int64_t* __restrict__ counts, int width, T divisor, int shift, int vec)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    int64_t const r = blockIdx.x;
    T const* srow = src + r * src_stride_row;
    if (blockIdx.y == 0)
    {
        count_row<T>(srow, src_stride_col, width, counts + r);
        return;
    }
    decode_chunk<T>(srow, src_stride_col, tables + r * tables_stride, width, static_cast<int>(blockIdx.y) - 1, divisor,
        shift, vec != 0);
}

int pow2_shift(int64_t d)
{
    if (d <= 0 || (d & (d - 1)) != 0)
        return -1;
    int s = 0;
    while ((int64_t(1) << s) != d)
        ++s;
    return s;
}

bool aligned16(void const* p)
{
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

template <typename T>
int launch_t(void const* src, int64_t sr, int64_t sc, void* tables, int64_t ts, void* counts, int rows, int width,
    T divisor, int shift, int pdl, int wpc, void* stream)
{
    cudaLaunchConfig_t cfg = {};
    cfg.stream = static_cast<cudaStream_t>(stream);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    if (width > kTinyWidth / 2 && width <= kTinyWidth)
    {
        cfg.gridDim = dim3((rows + kE16W2Rows - 1) / kE16W2Rows);
        cfg.blockDim = dim3(64 * (rows < kE16W2Rows ? rows : kE16W2Rows));
        return static_cast<int>(cudaLaunchKernelEx(&cfg, dspark_refresh_tiny_w2_kernel<T>, static_cast<T const*>(src),
            sr, sc, static_cast<int32_t*>(tables), ts, static_cast<int64_t*>(counts), rows, width, divisor, shift));
    }
    if (width <= kTinyWidth / 2)
    {
        cfg.gridDim = dim3((rows + kTinyRows - 1) / kTinyRows);
        cfg.blockDim = dim3(32 * (rows < kTinyRows ? rows : kTinyRows));
        int const per = (width + 31) / 32; // 0 .. 8
        auto* const kernel = per <= 1 ? dspark_refresh_tiny_kernel<T, 1>
            : per <= 2                ? dspark_refresh_tiny_kernel<T, 2>
            : per <= 4                ? dspark_refresh_tiny_kernel<T, 4>
                                      : dspark_refresh_tiny_kernel<T, 8>;
        return static_cast<int>(cudaLaunchKernelEx(&cfg, kernel, static_cast<T const*>(src), sr, sc,
            static_cast<int32_t*>(tables), ts, static_cast<int64_t*>(counts), rows, width, divisor, shift));
    }
    int const nthr = 128 * wpc;
    int const chunk = 4 * nthr;
    int const chunks = (width + chunk - 1) / chunk;
    if (chunks + 1 > 65535)
        return -4;
    bool const vec = sc == 1 && aligned16(src) && aligned16(tables)
        && (rows == 1 || ((sr * static_cast<int64_t>(sizeof(T))) % 16 == 0 && (ts * 4) % 16 == 0));
    cfg.gridDim = dim3(rows, chunks + 1);
    cfg.blockDim = dim3(nthr);
    return static_cast<int>(cudaLaunchKernelEx(&cfg, dspark_refresh_wide_kernel<T>, static_cast<T const*>(src), sr, sc,
        static_cast<int32_t*>(tables), ts, static_cast<int64_t*>(counts), width, divisor, shift, vec ? 1 : 0));
}
} // namespace

namespace
{
constexpr int kLaunchWpc = 4; // 128 x 4 = 512 threads per wide CTA (the measured p1w4 arm)
static_assert(kDsparkRefreshMaxWidth <= 65534 * 4 * 128 * kLaunchWpc, "the wide grid's y dimension");
} // namespace

void invokeDsparkRefreshBlockTables(void const* src, bool srcIs64, int64_t srcStrideRow, int64_t srcStrideCol,
    int32_t* tables, int64_t tablesStride, int64_t* counts, int32_t rows, int32_t width, int64_t divisor,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(divisor >= 1 && width >= 0 && width <= kDsparkRefreshMaxWidth,
        "dspark refresh block tables: bad divisor / width");
    if (rows <= 0)
        return;
    int shift = pow2_shift(divisor);
    int rc = 0;
    if (!srcIs64)
    {
        if (divisor > 2147483647LL)
            shift = 31; // every clamped int32 value is < 2^31 <= divisor: quotient 0, as the int64 decode
        rc = launch_t<int32_t>(src, srcStrideRow, srcStrideCol, tables, tablesStride, counts, rows, width,
            static_cast<int32_t>(shift >= 0 ? 1 : divisor), shift, 1, kLaunchWpc, stream);
    }
    else
    {
        rc = launch_t<int64_t>(src, srcStrideRow, srcStrideCol, tables, tablesStride, counts, rows, width, divisor,
            shift, 1, kLaunchWpc, stream);
    }
    TLLM_CHECK_WITH_INFO(rc >= 0, "dspark refresh block tables: shape past the launch grid");
    TLLM_CUDA_CHECK(static_cast<cudaError_t>(rc));
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
