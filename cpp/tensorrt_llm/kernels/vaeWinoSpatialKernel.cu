/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/packedVector.cuh"
#include "vaeNormCommon.cuh"
#include "vaeWinoSpatialKernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

using common::PackedAligned;
using namespace vae_norm;

constexpr int kThreads = 256;
// Frames are bf16 or fp16 (template parameter T); the transformed data (V, M) is fp16, which keeps
// three more mantissa bits than bf16 through the GEMMs.
using TS = __half;

// Channels per thread: 4 (8-byte vectors) for m = 2; 2 (one bf16x2 / fp16x2 pair) for m = 4,
// whose 6x6 window of fp32 values has to stay in registers.
template <typename T, int M>
struct Lanes
{
    static constexpr int V = M == 2 ? 4 : 2;
    using Vec = PackedAligned<T, V>;
    using VecS = PackedAligned<TS, V>;
};

struct Geometry
{
    int tw;                // tiles per row (ceil(W / m))
    int stripRows;         // row1 - row0
    int64_t tilesPerFrame; // stripRows * tw
    int64_t tilesPerComp;  // pairs * tilesPerFrame
    int pairs;
    int64_t coordStride;   // elements between consecutive coordinates of V / M
};

template <int M>
__device__ __forceinline__ Geometry geometry(VaeWinoSpatialParams const& p)
{
    Geometry g;
    g.tw = (p.width + M - 1) / M;
    g.stripRows = p.row1 - p.row0;
    g.tilesPerFrame = static_cast<int64_t>(g.stripRows) * g.tw;
    g.pairs = p.frames / p.components;
    g.tilesPerComp = g.pairs * g.tilesPerFrame;
    g.coordStride = p.stackFrames ? g.tilesPerFrame * p.frames * p.channels : g.tilesPerComp * p.channels;
    return g;
}

// Element offset of coordinate 0 of V / M for frame n (component comp, pair within it), a tile
// and channel c0. Layout [comp * coords, pairs * tiles, C], or with stacked frames
// [coords, tiles, N * C] (frame n at channel offset n * C).
template <int COORDS>
__device__ __forceinline__ int64_t vmOffset(
    Geometry const& g, VaeWinoSpatialParams const& p, int n, int comp, int pair, int64_t tile, int c0)
{
    if (p.stackFrames)
    {
        return tile * (static_cast<int64_t>(p.frames) * p.channels) + static_cast<int64_t>(n) * p.channels + c0;
    }
    int64_t const row = static_cast<int64_t>(comp) * COORDS * g.tilesPerComp + pair * g.tilesPerFrame + tile;
    return row * p.channels + c0;
}

// Base of frame n (channel 0): the `frames` buffer, or the segment holding the frame; nullptr
// for a zero frame.
template <typename T>
__device__ __forceinline__ T const* frameBase(T const* frames, VaeWinoSpatialParams const& p, int n)
{
    int64_t const frameElems = static_cast<int64_t>(p.height) * p.width * p.channels;
    if (p.segments == 0)
    {
        return frames + n * frameElems;
    }
    int begin = 0;
#pragma unroll
    for (int s = 0; s < 3; ++s)
    {
        if (s < p.segments && n < begin + p.segFrames[s])
        {
            return p.segPtr[s] == nullptr
                ? nullptr
                : static_cast<T const*>(p.segPtr[s]) + static_cast<int64_t>(n - begin) * frameElems;
        }
        begin += s < p.segments ? p.segFrames[s] : 0;
    }
    return nullptr;
}

__device__ __forceinline__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float2 operator*(float k, float2 a)
{
    return make_float2(k * a.x, k * a.y);
}

// Winograd transforms written out. F(2x2, 3x3): Lavin & Gray, points (0, +-1, inf), integer
// coefficients. F(4x4, 3x3): points (0, +-3/2, +-5/8, inf), the same set as the sm100 CuTe DSL
// path: the real point set with the lowest fp16 rounding sensitivity of the stored V / M (1.7e-3
// per conv against 2.5e-3 for (0, +-1, 2, -1/2, inf)). Its B^T and A^T entries are dyadic, so
// every product with a bf16 / fp16 input is exact in fp32 and only the sums round. The +-p rows
// share an even and an odd part.
// bt<M>(x, d): row x of B^T applied to the (m+2) values d; at<M>(r, t): row r of A^T.
template <int M>
struct WinoMath;

template <>
struct WinoMath<2>
{
    static constexpr int A = 4;

    static __device__ __forceinline__ float2 bt(int x, float2 const (&d)[4])
    {
        switch (x)
        {
        case 0: return d[0] - d[2];
        case 1: return d[1] + d[2];
        case 2: return d[2] - d[1];
        default: return d[1] - d[3];
        }
    }

    static __device__ __forceinline__ float2 at(int r, float2 const (&t)[4])
    {
        return r == 0 ? t[0] + t[1] + t[2] : t[1] - t[2] - t[3];
    }
};

template <>
struct WinoMath<4>
{
    static constexpr int A = 6;

    static __device__ __forceinline__ float2 bt(int x, float2 const (&d)[6])
    {
        switch (x)
        {
        case 0: return 0.87890625f * d[0] - 2.640625f * d[2] + d[4];
        case 1: return (d[4] - 0.390625f * d[2]) + (1.5f * d[3] - 0.5859375f * d[1]);
        case 2: return (d[4] - 0.390625f * d[2]) - (1.5f * d[3] - 0.5859375f * d[1]);
        case 3: return (d[4] - 2.25f * d[2]) + (0.625f * d[3] - 1.40625f * d[1]);
        case 4: return (d[4] - 2.25f * d[2]) - (0.625f * d[3] - 1.40625f * d[1]);
        default: return 0.87890625f * d[1] - 2.640625f * d[3] + d[5];
        }
    }

    static __device__ __forceinline__ float2 at(int r, float2 const (&t)[6])
    {
        switch (r)
        {
        case 0: return t[0] + t[1] + t[2] + t[3] + t[4];
        case 1: return 1.5f * (t[1] - t[2]) + 0.625f * (t[3] - t[4]);
        case 2: return 2.25f * (t[1] + t[2]) + 0.390625f * (t[3] + t[4]);
        default: return 3.375f * (t[1] - t[2]) + 0.244140625f * (t[3] - t[4]) + t[5];
        }
    }
};

// V = B^T d B for one (m+2) x (m+2) window and V channels, written as (m+2)^2 vectors; one
// coordinate row at a time so the register footprint stays bounded.
template <typename T, int M>
__global__ void __launch_bounds__(kThreads)
    vaeWinoSpatialInputKernel(T const* __restrict__ frames, TS* __restrict__ v, VaeWinoSpatialParams p)
{
    using W = WinoMath<M>;
    constexpr int A = W::A;
    constexpr int V = Lanes<T, M>::V;
    using Vec = typename Lanes<T, M>::Vec;
    using VecS = typename Lanes<T, M>::VecS;
    Geometry const g = geometry<M>(p);
    int const C = p.channels;
    int const cvec = C / V;
    // Grid: x over (tile column, channel-vector block), y over the strip's tile rows, z over
    // frames (see gridFor); the per-block tile and frame indices are uniform, so no thread does
    // a 64-bit division.
    int const j = blockIdx.x % g.tw;
    int const cv = (blockIdx.x / g.tw) * kThreads + threadIdx.x;
    if (cv < cvec)
    {
        int const c0 = cv * V;
        int const i = p.row0 + blockIdx.y;
        int const n = blockIdx.z;
        int const tile = blockIdx.y * g.tw + j;
        int const comp = n / g.pairs, pair = n - comp * g.pairs;

        Vec d[A][A];
        T const* frame0 = frameBase(frames, p, n);
        bool const present = frame0 != nullptr;
#pragma unroll
        for (int a = 0; a < A; ++a)
        {
            int const row = M * i - 1 + a;
#pragma unroll
            for (int b = 0; b < A; ++b)
            {
                int const col = M * j - 1 + b;
                if (present && row >= 0 && row < p.height && col >= 0 && col < p.width)
                {
                    d[a][b]
                        = *reinterpret_cast<Vec const*>(frame0 + (static_cast<int64_t>(row) * p.width + col) * C + c0);
                }
                else
                {
                    d[a][b] = common::make_zeros<Vec>();
                }
            }
        }
        // Keep every load of the window in flight before the arithmetic starts: without the
        // barrier the compiler sinks each load to its single use and the tile pays one
        // memory latency per window column instead of one per window.
        asm volatile("" ::: "memory");
        TS* vp = v + vmOffset<A * A>(g, p, n, comp, pair, tile, c0);
        int64_t const coordStride = g.coordStride;
        // Two separable passes per channel pair e: T[x][b] = row x of B^T applied to window
        // column b, then V[x][y] = row y of B^T applied to row x of T. Every input value is
        // converted once and every transform row evaluated once.
        VecS out[A * A];
#pragma unroll
        for (int e = 0; e < V / 2; ++e)
        {
            float2 tr[A][A];
#pragma unroll
            for (int b = 0; b < A; ++b)
            {
                float2 col[A];
#pragma unroll
                for (int a = 0; a < A; ++a)
                {
                    col[a] = pairAt(d[a][b], e);
                }
#pragma unroll
                for (int x = 0; x < A; ++x)
                {
                    tr[x][b] = W::bt(x, col);
                }
            }
#pragma unroll
            for (int x = 0; x < A; ++x)
            {
#pragma unroll
                for (int y = 0; y < A; ++y)
                {
                    setPair(out[x * A + y], e, W::bt(y, tr[x]));
                }
            }
        }
#pragma unroll
        for (int k = 0; k < A * A; ++k)
        {
            *reinterpret_cast<VecS*>(vp) = out[k];
            vp += coordStride;
        }
    }
}

// Staged input transform: a block covers one tile row x kStagedTiles consecutive tiles x
// kStagedChannels channels of one frame. The block first copies the (m+2) x (m * kStagedTiles + 2)
// window rows it needs into shared memory with 16-byte loads (zero outside the frame), then
// every thread transforms one (tile, channel pair) from shared memory, where all window
// addresses are compile-time offsets from the thread's base. Needs channels % kStagedChannels == 0.
constexpr int kStagedTiles = 8;
constexpr int kStagedChannels = 64;
static_assert(kStagedTiles * (kStagedChannels / 2) == kThreads, "one thread per (tile, channel pair)");

template <typename T, int M>
__global__ void __launch_bounds__(kThreads)
    vaeWinoSpatialInputStagedKernel(T const* __restrict__ frames, TS* __restrict__ v, VaeWinoSpatialParams p)
{
    using W = WinoMath<M>;
    constexpr int A = W::A;
    constexpr int Cols = M * kStagedTiles + 2;
    constexpr int CH = kStagedChannels;
    using PairS = PackedAligned<TS, 2>; // one channel pair per thread, whatever Lanes<T, M>::V is
    __shared__ __align__(16) T win[A][Cols][CH];

    Geometry const g = geometry<M>(p);
    int const C = p.channels;
    // Grid: x over (tile-column group, channel chunk), y over the strip's tile rows, z over frames.
    int const groupsPerRow = (g.tw + kStagedTiles - 1) / kStagedTiles;
    int const chunk = blockIdx.x / groupsPerRow;
    int const i = p.row0 + blockIdx.y;
    int const j0 = (blockIdx.x % groupsPerRow) * kStagedTiles;
    int const n = blockIdx.z;
    int const comp = n / g.pairs, pair = n - comp * g.pairs;
    int const row0 = M * i - 1, col0 = M * j0 - 1;

    // Copy phase: 16-byte cp.async chunks over (window row, window column, 8-channel chunk),
    // all issued before the wait; chunks outside the frame are zero-filled by the copy unit.
    {
        constexpr int ChunksPerPixel = CH / 8;
        constexpr int Total = A * Cols * ChunksPerPixel;
        constexpr int Iters = (Total + kThreads - 1) / kThreads;
        T const* frame0 = frameBase(frames, p, n);
        bool const present = frame0 != nullptr;
        // A zero frame has no source; the copies below then move 0 bytes from a valid address.
        T const* base = present ? frame0 + chunk * CH : reinterpret_cast<T const*>(v);
#pragma unroll
        for (int it = 0; it < Iters; ++it)
        {
            int const q = threadIdx.x + it * kThreads;
            if (q < Total)
            {
                int const cc = q % ChunksPerPixel;
                int const pix = q / ChunksPerPixel;
                int const a = pix / Cols, b = pix - a * Cols;
                int const row = row0 + a, col = col0 + b;
                bool const in = present && row >= 0 && row < p.height && col >= 0 && col < p.width;
                T const* src = in ? base + (static_cast<int64_t>(row) * p.width + col) * C + cc * 8 : base;
                unsigned const dst = static_cast<unsigned>(__cvta_generic_to_shared(&win[a][b][cc * 8]));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(in ? 16 : 0));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
    }
    __syncthreads();

    // Transform phase: thread -> (tile within the group, channel pair).
    int const tt = threadIdx.x / (CH / 2);
    int const pp = threadIdx.x - tt * (CH / 2);
    int const j = j0 + tt;
    if (j >= g.tw)
    {
        return;
    }
    float2 tr[A][A]; // T[x][b]
#pragma unroll
    for (int b = 0; b < A; ++b)
    {
        float2 col[A];
#pragma unroll
        for (int a = 0; a < A; ++a)
        {
            col[a] = Elem<T>::toFloat2(*reinterpret_cast<typename Elem<T>::Pair const*>(&win[a][M * tt + b][2 * pp]));
        }
#pragma unroll
        for (int x = 0; x < A; ++x)
        {
            tr[x][b] = W::bt(x, col);
        }
    }
    int const tile = (i - p.row0) * g.tw + j;
    TS* vp = v + vmOffset<A * A>(g, p, n, comp, pair, tile, chunk * CH + 2 * pp);
    int64_t const coordStride = g.coordStride;
#pragma unroll
    for (int x = 0; x < A; ++x)
    {
#pragma unroll
        for (int y = 0; y < A; ++y)
        {
            PairS out;
            setPair(out, 0, W::bt(y, tr[x]));
            *reinterpret_cast<PairS*>(vp) = out;
            vp += coordStride;
        }
    }
}

// y = A^T M A for one tile and V channels: (m+2)^2 vectors in, m x m output pixels out.
template <typename T, int M>
__global__ void __launch_bounds__(kThreads) vaeWinoSpatialOutputKernel(
    TS const* __restrict__ m, T* __restrict__ frames, T const* __restrict__ bias, VaeWinoSpatialParams p)
{
    using W = WinoMath<M>;
    constexpr int A = W::A;
    constexpr int V = Lanes<T, M>::V;
    using Vec = typename Lanes<T, M>::Vec;
    using VecS = typename Lanes<T, M>::VecS;
    Geometry const g = geometry<M>(p);
    int const C = p.channels;
    int const cvec = C / V;
    // Grid as in the input kernel (see gridFor).
    int const j = blockIdx.x % g.tw;
    int const cv = (blockIdx.x / g.tw) * kThreads + threadIdx.x;
    if (cv < cvec)
    {
        int const c0 = cv * V;
        int const i = p.row0 + blockIdx.y;
        int const n = blockIdx.z;
        int const tile = blockIdx.y * g.tw + j;
        int const comp = n / g.pairs, pair = n - comp * g.pairs;

        VecS mv[A * A];
        TS const* mp = m + vmOffset<A * A>(g, p, n, comp, pair, tile, c0);
        int64_t const coordStride = g.coordStride;
#pragma unroll
        for (int k = 0; k < A * A; ++k)
        {
            mv[k] = *reinterpret_cast<VecS const*>(mp);
            mp += coordStride;
        }
        Vec const bv = bias ? *reinterpret_cast<Vec const*>(bias + c0) : common::make_zeros<Vec>();
        asm volatile("" ::: "memory"); // all coordinate loads in flight before the arithmetic
        T* frame0 = frames + (static_cast<int64_t>(n) * p.height * p.width) * C + c0;
        // Two separable passes per channel pair e: T[r][b] = row r of A^T applied to
        // coordinate column b, then y[r][s] = row s of A^T applied to row r of T.
        Vec out[M * M];
#pragma unroll
        for (int e = 0; e < V / 2; ++e)
        {
            float2 tr[M][A];
#pragma unroll
            for (int b = 0; b < A; ++b)
            {
                float2 col[A];
#pragma unroll
                for (int x = 0; x < A; ++x)
                {
                    col[x] = pairAt(mv[x * A + b], e);
                }
#pragma unroll
                for (int r = 0; r < M; ++r)
                {
                    tr[r][b] = W::at(r, col);
                }
            }
            float2 const bp = pairAt(bv, e);
#pragma unroll
            for (int r = 0; r < M; ++r)
            {
#pragma unroll
                for (int sidx = 0; sidx < M; ++sidx)
                {
                    setPair(out[r * M + sidx], e, W::at(sidx, tr[r]) + bp);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < M; ++r)
        {
            T* rowp = frame0 + (static_cast<int64_t>(M * i + r) * p.width + M * j) * C;
#pragma unroll
            for (int sidx = 0; sidx < M; ++sidx)
            {
                if (M * i + r < p.height && M * j + sidx < p.width)
                {
                    *reinterpret_cast<Vec*>(rowp + sidx * C) = out[r * M + sidx];
                }
            }
        }
    }
}

// Grid of the plain input and the output kernels: x = tile columns x channel-vector blocks
// (column fastest), y = the strip's tile rows, z = frames. Only x can exceed 65535; y and z are
// bounded by the frame height and the frame count.
dim3 gridFor(VaeWinoSpatialParams const& p)
{
    int const lanes = p.tile == 2 ? Lanes<__nv_bfloat16, 2>::V : Lanes<__nv_bfloat16, 4>::V;
    int const cvec = p.channels / lanes;
    int const tw = (p.width + p.tile - 1) / p.tile;
    int const rows = p.row1 - p.row0;
    TLLM_CHECK_WITH_INFO(
        rows <= 65535 && p.frames <= 65535, "vaeWinoSpatial: too many tile rows or frames for the grid");
    return dim3(tw * ((cvec + kThreads - 1) / kThreads), rows, p.frames);
}

void check(VaeWinoSpatialParams const& p)
{
    TLLM_CHECK_WITH_INFO(p.tile == 2 || p.tile == 4, "vaeWinoSpatial: tile must be 2 or 4");
    TLLM_CHECK_WITH_INFO(p.channels % 8 == 0, "vaeWinoSpatial: channels must be a multiple of 8");
    TLLM_CHECK_WITH_INFO(
        p.components >= 1 && p.frames % p.components == 0, "vaeWinoSpatial: frames must be a multiple of components");
    TLLM_CHECK_WITH_INFO(!p.stackFrames || p.components == 1, "vaeWinoSpatial: stacked frames need components == 1");
    TLLM_CHECK_WITH_INFO(0 <= p.row0 && p.row0 < p.row1 && p.row1 <= (p.height + p.tile - 1) / p.tile,
        "vaeWinoSpatial: bad tile-row range");
    TLLM_CHECK_WITH_INFO(0 <= p.segments && p.segments <= 3, "vaeWinoSpatial: at most three frame segments");
    if (p.segments > 0)
    {
        TLLM_CHECK_WITH_INFO(p.stackFrames, "vaeWinoSpatial: frame segments need stacked frames");
        int total = 0;
        for (int s = 0; s < p.segments; ++s)
        {
            TLLM_CHECK_WITH_INFO(p.segFrames[s] > 0, "vaeWinoSpatial: empty frame segment");
            total += p.segFrames[s];
        }
        TLLM_CHECK_WITH_INFO(total == p.frames, "vaeWinoSpatial: frame segments must add up to N");
    }
}

template <typename T, int M>
void launchInputStaged(T const* frames, TS* v, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    int const tw = (p.width + M - 1) / M;
    int const groupsPerRow = (tw + kStagedTiles - 1) / kStagedTiles;
    int const rows = p.row1 - p.row0;
    TLLM_CHECK_WITH_INFO(
        rows <= 65535 && p.frames <= 65535, "vaeWinoSpatial: too many tile rows or frames for the grid");
    dim3 const grid(groupsPerRow * (p.channels / kStagedChannels), rows, p.frames);
    vaeWinoSpatialInputStagedKernel<T, M><<<grid, kThreads, 0, stream>>>(frames, v, p);
}

template <typename T>
void launchInputTyped(void const* frames, void* v, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    T const* f = static_cast<T const*>(frames);
    TS* out = static_cast<TS*>(v);
    if (p.channels % kStagedChannels == 0)
    {
        p.tile == 2 ? launchInputStaged<T, 2>(f, out, p, stream) : launchInputStaged<T, 4>(f, out, p, stream);
    }
    else
    {
        dim3 const grid = gridFor(p);
        if (p.tile == 2)
        {
            vaeWinoSpatialInputKernel<T, 2><<<grid, kThreads, 0, stream>>>(f, out, p);
        }
        else
        {
            vaeWinoSpatialInputKernel<T, 4><<<grid, kThreads, 0, stream>>>(f, out, p);
        }
    }
}

template <typename T>
void launchOutputTyped(
    void const* m, void* frames, void const* bias, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    dim3 const grid = gridFor(p);
    if (p.tile == 2)
    {
        vaeWinoSpatialOutputKernel<T, 2><<<grid, kThreads, 0, stream>>>(
            static_cast<TS const*>(m), static_cast<T*>(frames), static_cast<T const*>(bias), p);
    }
    else
    {
        vaeWinoSpatialOutputKernel<T, 4><<<grid, kThreads, 0, stream>>>(
            static_cast<TS const*>(m), static_cast<T*>(frames), static_cast<T const*>(bias), p);
    }
}

} // namespace

void launchVaeWinoSpatialInput(void const* frames, void* v, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    check(p);
    if (p.halfFrames)
        launchInputTyped<__half>(frames, v, p, stream);
    else
        launchInputTyped<__nv_bfloat16>(frames, v, p, stream);
    sync_check_cuda_error(stream);
}

void launchVaeWinoSpatialOutput(
    void const* m, void* frames, void const* bias, VaeWinoSpatialParams const& p, cudaStream_t stream)
{
    check(p);
    TLLM_CHECK_WITH_INFO(!p.stackFrames, "vaeWinoSpatial: the output transform has no stacked-frame layout");
    if (p.halfFrames)
        launchOutputTyped<__half>(m, frames, bias, p, stream);
    else
        launchOutputTyped<__nv_bfloat16>(m, frames, bias, p, stream);
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
