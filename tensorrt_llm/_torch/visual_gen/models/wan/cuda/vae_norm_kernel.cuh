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

// The Wan VAE position-walk kernel: one thread group (L lanes, K vectors each) owns one
// spatial position of a batch-1 channels-last video ([T, P, C], rows contiguous) and walks
// its frames in order. Three compile-time stages make it every norm-side kernel of the VAE:
//
//   input  (IN_WINO)   x frames, or the temporal Winograd combine of order + 2 conv products
//                      (frame g*R + j is sum_k AT[j][k] m_k[g], index order), then the deferred
//                      conv bias (inBias) and the residual; the pre-norm row goes to ``sum``
//                      when that pointer is set.
//   middle (MID)       0: nothing (the combine kernel); 1: RMSNorm; 2: RMSNorm + SiLU.
//   output (OUT)       0: the temporal Winograd slabs of order R (D_k = sum_i BT[k][i] f_i over
//                      the group's R + 2 normalized frames, two of them the cache), with the
//                      last two normalized frames written to ``yTail`` for the next chunk;
//                      1: plain rows to ``y`` ([T, P', C], P' the padded position count when
//                      PADDED: the downsampler's zero row/column appended).
//
// Numerics: fp32 throughout, one rounding per output unless REF (reference rounding), which
// reproduces the eager bf16 rounding after every add, normalize, multiply and bias. The
// order-2 bf16 slab path adds no rounding beyond the F(2,3) arithmetic itself. Compiled at run
// time per instantiation (jit_norm.py builds the launch); the tables live in vae_wino_tables.cuh.

#pragma once

#include "packed_vector.cuh"
#include "vae_norm_common.cuh"
#include "vae_wino_tables.cuh"
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace vaewino
{

using cuda::std::int64_t;
using common::PackedAligned;
using namespace vae_norm;
using namespace vae_wino;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxComponents = 6;

enum Mid : int
{
    kMidNone = 0,
    kMidNorm = 1,
    kMidNormSilu = 2
};

enum Out : int
{
    kOutSlabs = 0,
    kOutRows = 1
};

struct VaeNormParams
{
    int64_t positions; // P: source positions (H * W)
    int channels;      // C (multiple of 8)
    int frames;        // T, a positive multiple of the walk's group size
    int cacheFrames;   // 0, 1 or 2 (slab output only)
    int height;        // H and W of a frame (padded row output only)
    int width;
    int padH;          // extra zero rows at the bottom of the row output (0 or 1)
    int padW;          // extra zero columns at the right (0 or 1)
    // Physical frame geometry of the products (IN_MAP) and of the slab / row output (OUT_MAP)
    // when they carry halo rows or columns: logical position (i, j) of a W-wide frame sits at
    // i * physW + j + physOff inside a frame of physFrame positions.
    long long inFrame;
    int inW;
    int inOff;
    long long outFrame;
    int outW;
    int outOff;
    // Upsample input (UP_FT > 0): the frames are the phase gather of the packed 2x2 upsample
    // conv output ``x`` [T, (upHeight+1) * (upWidth+1), 4C] plus the DupUp3D shortcut
    // ``residual`` [framesR, upHeight * upWidth, upCin] (upCrop leading frames dropped).
    int upHeight;
    int upWidth;
    int upCrop;
    int upCin;
    float scale;
    float eps;
    // Halo push (HALO): the output's first and last logical line along the split axis are
    // stored a second time, into the neighbouring ranks' buffers (their halo lines). haloAxis
    // 0: the frames are split along their height (rows travel), 1: along their width
    // (columns travel). Per side (L: the lower rank, R: the higher; null without a
    // neighbour): the peer's address of the destination line's first position (frame 0,
    // component 0), its positions per frame, the positions between consecutive line
    // positions and the elements between its slab components.
    void* haloL;
    void* haloR;
    long long haloFrameL;
    long long haloFrameR;
    long long haloCompL;
    long long haloCompR;
    int haloStepL;
    int haloStepR;
    int haloAxis;
};

// T: activations (x, residual, cache, sum, yTail, y, gamma, bias); TD: slabs d and products m,
// each one stacked buffer [components, groups, P, C] with compStride elements per component.
template <typename T, typename TD>
struct Ptrs
{
    T const* x;
    TD const* m;
    T const* residual;
    T const* inBias;
    T const* cache;
    T const* gamma;
    T const* bias;
    T* sum;
    T* yTail;
    T* y;
    TD* d;
    int64_t compStride;   // elements between the slab components of d
    int64_t inCompStride; // elements between the product components of m
};

// a (+/-) b with the eager bf16 rounding in reference mode.
template <typename T, bool REF>
__device__ __forceinline__ float addR(float a, float b)
{
    float const s = a + b;
    return REF ? roundTo<T>(s) : s;
}

// Resident blocks per SM the register budget is sized for. Per-thread state is bounded
// independently of K: the row being normalized (K vectors) plus two vector slices of
// inputs in flight; the frame history lives in shared memory. Order 4 holds up to five
// products per slice and emits five slabs from one frame: at the order-2 budgets it spills
// (measured: 11-19 M local loads per launch, DRAM 44-62%), so it runs two blocks per SM.
// The row-output modes carry no ring and no slabs and keep four blocks up to K = 2.
#ifndef VAEWINO_IN_BLOCKS
#define VAEWINO_IN_BLOCKS 0 // 0: the default below; else blocks per SM for the order-4 product combine
#endif
#ifndef VAEWINO_IN_DEPTH
#define VAEWINO_IN_DEPTH 0 // 0: the default below; else slice pipeline depth for the product combine
#endif

template <int K, bool IN_WINO, int R, int OUT>
__host__ __device__ constexpr int blocksPerSm()
{
    if (OUT == kOutRows)
    {
        return K <= 2 ? 4 : 3;
    }
    if (IN_WINO && R == 4 && VAEWINO_IN_BLOCKS != 0)
    {
        return VAEWINO_IN_BLOCKS;
    }
    if (K >= 4 || R == 4)
    {
        return 2;
    }
    return (!IN_WINO && K <= 2) ? 4 : 3;
}

// Vector slices of one frame kept in flight ahead of the one being normalized (1: the slice
// is loaded right before it is used; the product combine holds five vectors per slice, so a
// shallower pipeline trades latency hiding for registers).
template <int K, bool IN_WINO>
__host__ __device__ constexpr int sliceDepth()
{
    if (IN_WINO && VAEWINO_IN_DEPTH != 0)
    {
        return VAEWINO_IN_DEPTH;
    }
    return K >= 4 ? 3 : 2;
}

// History ring: a group needs frames t-2 .. t+R-1; frame t-2 is dead once the slabs that
// read it are out (while frame t+R-2 is normalized), so frame t+R-1 can take its slot:
// R + 1 slots suffice. slot(frame) = (frame + 2) mod (R + 1); the cached frames -2, -1
// start in slots 0 and 1.
template <int R>
__host__ __device__ constexpr int ringSlots()
{
    return R + 1;
}

template <int R>
__host__ __device__ __forceinline__ int ringSlot(int frame)
{
    return (frame + 2) % ringSlots<R>();
}

template <int L>
__host__ __device__ constexpr int positionsPerBlock()
{
    return (32 / L) * kWarps;
}

template <int L, int R, int OUT>
__host__ __device__ constexpr int sharedBytes(int channels)
{
    return 3 * channels * 4 + (OUT == kOutSlabs ? ringSlots<R>() * positionsPerBlock<L>() * channels * 2 : 0);
}

#ifndef VAEWINO_XF_IN
#define VAEWINO_XF_IN 0
#endif

template <typename T, typename TD, int R, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS,
    bool HAS_RESIDUAL, bool IN_WINO, int MID, int OUT, bool PADDED, bool IN_MAP, bool OUT_MAP, int UP_FT, int UP_RATIO,
    bool HALO>
__device__ __forceinline__ void vaeNormBody(Ptrs<T, TD> const& ptrs, VaeNormParams const& p)
{
    constexpr bool SLABS = OUT == kOutSlabs;
    static_assert(!HALO || (OUT_MAP && !PADDED), "the halo push needs the output's frame geometry");
    // The frames come from the phase gather of a packed 2x2 upsample conv output (plus the
    // DupUp3D shortcut as the residual): UP_FT is the shortcut's temporal factor, UP_RATIO the
    // number of shortcut elements per output channel (factor / repeats).
    constexpr bool IN_UP = UP_FT > 0;
    static_assert(!IN_UP || (!IN_WINO && !IN_MAP && !PADDED), "the upsample input is a plain-frame source");
    static_assert(UP_FT >= 0 && UP_FT <= 2 && (UP_RATIO == 1 || UP_RATIO == 2), "unsupported upsample mapping");
    static_assert(!SLABS || MID == kMidNormSilu, "slabs are formed from normalized, activated frames");
    static_assert(!PADDED || OUT == kOutRows, "padding applies to the row output");
    static_assert(!(PADDED && OUT_MAP), "the padded row output has its own position map");
    static_assert(!IN_MAP || IN_WINO, "the input map applies to the products");
    static_assert(IN_WINO || SLABS || R == 1, "R is 1 unless products are combined or slabs are emitted");
    using Vec = PackedAligned<T, V>;
    using VecD = PackedAligned<TD, V>;
    using VecIn = cuda::std::conditional_t<IN_WINO, VecD, Vec>;
    using VecF = PackedAligned<float, V>;
    using W = Wino<(R < 2 ? 2 : R)>;
    constexpr int NUM_IN = IN_WINO ? W::kMaxTerms : 1;
    constexpr bool FRAME_PF = (K == 1);   // prefetch whole frames
    constexpr bool CONST_REGS = (K == 1); // per-channel constants in registers
    constexpr bool ACT = MID == kMidNormSilu;
    // The standalone norm op rounds the deferred-bias add and the residual add to bf16 in
    // every mode (the sum row must be the eager ``x + residual`` bit for bit); the slab modes
    // keep fp32 there unless REF.
    constexpr bool ROUND1 = REF || OUT == kOutRows || IN_UP;
    constexpr int kPosPerWarp = 32 / L;
    constexpr int kPosPerBlock = kPosPerWarp * kWarps;

    extern __shared__ __align__(16) unsigned char smemRaw[];
    float* const sGamma = reinterpret_cast<float*>(smemRaw);
    float* const sBias = sGamma + p.channels;
    float* const sInBias = sGamma + 2 * p.channels;
    T* const sHist = reinterpret_cast<T*>(sGamma + 3 * p.channels); // [slots][positions in block][C]

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const group = lane / L;
    int const laneInGroup = lane - group * L;
    int const C = p.channels;
    int64_t const P = p.positions;
    int const outW = p.width + p.padW;
    int64_t const POut = PADDED ? static_cast<int64_t>(p.height + p.padH) * outW : P;
    int64_t const inFrame = IN_MAP ? p.inFrame : P;       // physical positions per product frame
    int64_t const outFrame = OUT_MAP ? p.outFrame : POut; // physical positions per output frame
    // Physical index of a logical position inside a product / output frame.
    auto inIndex = [&](int64_t pos) -> int64_t
    {
        if constexpr (IN_MAP)
        {
            int const i = static_cast<int>(pos) / p.width;
            return static_cast<int64_t>(i) * p.inW + (static_cast<int>(pos) - i * p.width) + p.inOff;
        }
        else
        {
            return pos;
        }
    };
    auto outIndex = [&](int64_t pos) -> int64_t
    {
        if constexpr (OUT_MAP)
        {
            int const i = static_cast<int>(pos) / p.width;
            return static_cast<int64_t>(i) * p.outW + (static_cast<int>(pos) - i * p.width) + p.outOff;
        }
        else
        {
            return pos;
        }
    };
    int const posInBlock = warp * kPosPerWarp + group;

    // Table layout: the V floats of vector vi are split into 16-byte quarters stored in
    // separate planes ((q * nvec + vi) * 4 + e), so a warp reads each quarter with a
    // 16-byte lane stride (bank-conflict free; the natural 32-byte stride is 2-way conflicted).
    constexpr int kQuarters = V / 4;
    int const nvec = C / V;
    for (int c = tid; c < C; c += kThreads)
    {
        float const g = MID != kMidNone ? Elem<T>::toFloat(ptrs.gamma[c]) : 0.0f;
        int const idx = (((c % V) / 4) * nvec + c / V) * 4 + c % 4;
        sGamma[idx] = REF ? g : g * (ACT ? 0.5f * p.scale : p.scale);
        sBias[idx] = HAS_BIAS ? Elem<T>::toFloat(ptrs.bias[c]) : 0.0f;
        sInBias[idx] = HAS_IN_BIAS ? Elem<T>::toFloat(ptrs.inBias[c]) : 0.0f;
    }
    __syncthreads();

    int const laneChunkOff = laneInGroup * V;
    int64_t const gridPos = static_cast<int64_t>(gridDim.x) * kPosPerBlock;
    int const T_ = p.frames;

    auto tableVec = [&](float const* table, int k) -> VecF
    {
        VecF r;
        int const vi = laneInGroup + k * L;
#pragma unroll
        for (int q = 0; q < kQuarters; ++q)
        {
            float4 const f = *reinterpret_cast<float4 const*>(table + (q * nvec + vi) * 4);
            r.val[4 * q] = f.x;
            r.val[4 * q + 1] = f.y;
            r.val[4 * q + 2] = f.z;
            r.val[4 * q + 3] = f.w;
        }
        return r;
    };
    VecF gReg, bReg, ibReg;
    if constexpr (CONST_REGS)
    {
        gReg = tableVec(sGamma, 0);
        bReg = tableVec(sBias, 0);
        ibReg = tableVec(sInBias, 0);
    }
    auto constAt = [&](float const* table, VecF const& reg, int k) -> VecF
    {
        if constexpr (CONST_REGS)
        {
            return reg;
        }
        else
        {
            return tableVec(table, k);
        }
    };

    auto gRow = [&](auto const* base, int64_t row) { return base + row * C + laneChunkOff; };
    auto gRowW = [&](auto* base, int64_t row) { return base + row * C + laneChunkOff; };
    auto slotPtr = [&](int slot) { return sHist + (slot * kPosPerBlock + posInBlock) * C + laneChunkOff; };
    auto loadVec = [&](auto const* rowBase, int k)
    {
        using E = cuda::std::remove_cv_t<cuda::std::remove_pointer_t<decltype(rowBase)>>;
        return *reinterpret_cast<PackedAligned<E, V> const*>(rowBase + k * L * V);
    };
    auto storeVec = [&](auto* rowBase, int k, auto const& v) {
        *reinterpret_cast<cuda::std::remove_cv_t<cuda::std::remove_reference_t<decltype(v)>>*>(rowBase + k * L * V) = v;
    };

    // Source position of an output position (the row output may be padded); -1 for a pad.
    auto sourcePos = [&](int64_t opos) -> int64_t
    {
        if constexpr (!PADDED)
        {
            return opos;
        }
        else
        {
            int const oi = static_cast<int>(opos / outW), oj = static_cast<int>(opos - static_cast<int64_t>(oi) * outW);
            return (oi < p.height && oj < p.width) ? static_cast<int64_t>(oi) * p.width + oj : -1;
        }
    };

    // Row pointers of one frame's inputs: the x row, or the row inside component 0 of the
    // products (component k of a term sits compStride elements further; the term indices
    // are compile-time constants, so each load is one base plus a constant multiple).
    struct FrameSrc
    {
        cuda::std::conditional_t<IN_WINO, TD, T> const* in;
        T const* res;
        int resOff; // upsample input: the shortcut element of the lane's first output channel
    };

    // ``sposSrc``: the position's physical index inside a product frame (inIndex(spos)).
    auto srcOf = [&](int t, int64_t spos, int64_t sposSrc) -> FrameSrc
    {
        FrameSrc f{};
        int64_t const frow = static_cast<int64_t>(t) * P + spos;
        if constexpr (IN_UP)
        {
            // Output pixel (2i+pp, 2j+qq) of frame t reads temp[t, phase 2pp+qq, i+pp, j+qq]
            // and shortcut element (c * factor + S) / repeats = c * UP_RATIO + S / repeats of
            // low-res pixel (i, j) of frame (t + crop) / UP_FT, S = ((a*2+pp)*2+qq).
            int const w2 = 2 * p.upWidth;
            int const oi = static_cast<int>(spos / w2);
            int const oj = static_cast<int>(spos - static_cast<int64_t>(oi) * w2);
            int const i = oi >> 1, pp = oi & 1, j = oj >> 1, qq = oj & 1;
            int64_t const trow = (static_cast<int64_t>(t) * (p.upHeight + 1) + i + pp) * (p.upWidth + 1) + j + qq;
            f.in = ptrs.x + trow * (4 * C) + (2 * pp + qq) * C + laneChunkOff;
            if constexpr (HAS_RESIDUAL)
            {
                constexpr int REPEATS = 4 * UP_FT / UP_RATIO;
                int const ft = t + p.upCrop;
                int const tr = ft / UP_FT, a = ft - tr * UP_FT;
                int64_t const srow = (static_cast<int64_t>(tr) * p.upHeight + i) * p.upWidth + j;
                f.res = ptrs.residual + srow * p.upCin + static_cast<int64_t>(laneChunkOff) * UP_RATIO;
                f.resOff = ((a * 2 + pp) * 2 + qq) / REPEATS;
            }
            return f;
        }
        if constexpr (IN_WINO)
        {
            f.in = gRow(ptrs.m, static_cast<int64_t>(t / R) * inFrame + sposSrc);
        }
        else
        {
            f.in = gRow(ptrs.x, frow);
        }
        if constexpr (HAS_RESIDUAL)
        {
            f.res = gRow(ptrs.residual, frow);
        }
        return f;
    };

    // Two input buffers (named, never indexed, so they stay in registers): two frames
    // for FRAME_PF, otherwise two vector slices of one frame.
    VecIn inA[NUM_IN], inB[NUM_IN], inC[NUM_IN];
    Vec resA, resB, resC;
    // Loads slice k of frame J's inputs: the x row, or the products of the terms of AT row J.
    auto loadSlice = [&](auto jTag, VecIn(&dst)[NUM_IN], Vec& dres, FrameSrc const& f, int k)
    {
        constexpr int J = decltype(jTag)::value;
        if constexpr (IN_WINO)
        {
#pragma unroll
            for (int j = 0; j < numTerms<R>(J); ++j)
            {
                dst[j] = loadVec(f.in + termIndex<R>(J, j) * ptrs.inCompStride, k);
            }
        }
        else
        {
            dst[0] = loadVec(f.in, k);
        }
        if constexpr (HAS_RESIDUAL && IN_UP && UP_RATIO == 2)
        {
            // Output channel c reads shortcut element 2c + resOff: two vectors, every other
            // element, both parities built with constant indices and one selected.
            Vec const lo = *reinterpret_cast<Vec const*>(f.res + 2 * k * L * V);
            Vec const hi = *reinterpret_cast<Vec const*>(f.res + 2 * k * L * V + V);
            Vec even, odd;
#pragma unroll
            for (int e = 0; e < V; ++e)
            {
                even.val[e] = 2 * e < V ? lo.val[(2 * e) % V] : hi.val[(2 * e) % V];
                odd.val[e] = 2 * e + 1 < V ? lo.val[(2 * e + 1) % V] : hi.val[(2 * e + 1) % V];
            }
            dres = f.resOff == 0 ? even : odd;
        }
        else if constexpr (HAS_RESIDUAL)
        {
            dres = loadVec(f.res, k);
        }
    };

    // Cross-frame prefetch (sliceDepth 2, i.e. K <= 3): the first XF_SLICES slices of the
    // next frame (or of the next position's first frame) are loaded while this frame is
    // normalized, so a frame's load latency hides behind the previous frame's pass 2. One
    // input per slice prefetches two slices; the product combine holds up to five vectors per
    // slice, and more than VAEWINO_XF_IN slices in flight across pass 2 spills.
    constexpr int XF_SLICES = FRAME_PF || sliceDepth<K, IN_WINO>() > 2
        ? 0
        : (IN_WINO ? (VAEWINO_XF_IN > sliceDepth<K, IN_WINO>() ? sliceDepth<K, IN_WINO>() : VAEWINO_XF_IN) : 2);
    constexpr bool XF_PF = XF_SLICES > 0;
    constexpr bool XF_TWO = XF_SLICES > 1 && K > 1;
    constexpr bool PF = FRAME_PF || XF_PF;
    FrameSrc src{};
    bool prefetched = false; // the next position's first slices are already in inA/inB

    for (int64_t opos = static_cast<int64_t>(blockIdx.x) * kPosPerBlock + posInBlock; opos < POut; opos += gridPos)
    {
        int64_t const pos = sourcePos(opos);
        int64_t const posSrc = IN_MAP ? inIndex(pos) : pos;
        int64_t const posOut = OUT_MAP ? outIndex(opos) : opos;
        // The halo push: the neighbours this position's line also goes to (none for interior
        // positions; both when the shard is a single line) and the line position's row inside
        // each neighbour's frame 0.
        void* haloBaseL = nullptr;
        void* haloBaseR = nullptr;
        int64_t haloRowL = 0, haloRowR = 0;
        if constexpr (HALO)
        {
            int const i = static_cast<int>(pos) / p.width;
            int const j = static_cast<int>(pos) - i * p.width;
            int const across = p.haloAxis ? j : i;
            int const along = p.haloAxis ? i : j;
            int const n = p.haloAxis ? p.width : p.height;
            if (across == 0 && p.haloL != nullptr)
            {
                haloBaseL = p.haloL;
                haloRowL = static_cast<int64_t>(along) * p.haloStepL;
            }
            if (across == n - 1 && p.haloR != nullptr)
            {
                haloBaseR = p.haloR;
                haloRowR = static_cast<int64_t>(along) * p.haloStepR;
            }
        }
        if constexpr (PADDED)
        {
            if (pos < 0)
            {
                // A pad position: zero rows for every frame, nothing to read.
                for (int t = 0; t < T_; ++t)
                {
                    T* const dst = gRowW(ptrs.y, static_cast<int64_t>(t) * POut + opos);
#pragma unroll
                    for (int k = 0; k < K; ++k)
                    {
                        storeVec(dst, k, common::make_zeros<Vec>());
                    }
                }
                prefetched = false;
                continue;
            }
        }
        if constexpr (SLABS)
        {
            // Cached frames -2 and -1 into slots 0 and 1 (zeros where the cache is shorter).
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                int const cacheIdx = p.cacheFrames - 2 + h;
                T* const dst = slotPtr(ringSlot<R>(h - 2));
                if (cacheIdx >= 0)
                {
                    T const* const cs = gRow(ptrs.cache, static_cast<int64_t>(cacheIdx) * P + pos);
#pragma unroll
                    for (int k = 0; k < K; ++k)
                    {
                        storeVec(dst, k, loadVec(cs, k));
                    }
                }
                else
                {
#pragma unroll
                    for (int k = 0; k < K; ++k)
                    {
                        storeVec(dst, k, common::make_zeros<Vec>());
                    }
                }
            }
        }

        if (!prefetched)
        {
            src = srcOf(0, pos, posSrc);
            if constexpr (PF)
            {
                loadSlice(cuda::std::integral_constant<int, 0>{}, inA, resA, src, 0);
            }
            if constexpr (XF_TWO)
            {
                loadSlice(cuda::std::integral_constant<int, 0>{}, inB, resB, src, 1);
            }
        }
        prefetched = false;

        // One frame: J is its index within the group (compile time), t its frame number.
        // For FRAME_PF the inputs of frame J sit in buffer J % 2.
        auto frame = [&](auto jTag, int t)
        {
            constexpr int J = decltype(jTag)::value;
            constexpr int JN = (J + 1) % R;
            constexpr auto termsIn = cuda::std::integral_constant<int, J>{};
            constexpr auto termsNext = cuda::std::integral_constant<int, JN>{};
            // The frame that follows this one in the walk: the next frame of this position,
            // else frame 0 of this thread's next position (prefetch target).
            bool const nextInPos = t + 1 < T_;
            int64_t const nextPos = nextInPos ? pos : sourcePos(opos + gridPos);
            bool const hasNext = nextInPos || (opos + gridPos < POut && nextPos >= 0);
            int64_t const nextPosSrc = nextInPos ? posSrc : (IN_MAP && hasNext ? inIndex(nextPos) : nextPos);
            FrameSrc next{};
            if constexpr (FRAME_PF)
            {
                // The next frame's single slice is prefetched into inB and moved to inA once this
                // frame is done, so every frame reads inA whatever its parity.
                if (hasNext)
                {
                    next = srcOf(nextInPos ? t + 1 : 0, nextPos, nextPosSrc);
                    if (nextInPos)
                        loadSlice(termsNext, inB, resB, next, 0);
                    else
                        loadSlice(cuda::std::integral_constant<int, 0>{}, inB, resB, next, 0);
                }
            }
            else if constexpr (!XF_PF)
            {
                loadSlice(termsIn, inA, resA, src, 0);
            }

            // Pass 1: combine, deferred bias, residual; the bf16 row stays in registers (y)
            // and goes to ``sum``; the sum of squares is accumulated on the way. The input
            // buffer is chosen with a constant branch (never a selected reference) so both
            // buffers stay in registers.
            float ss = 0.0f;
            Vec y[K];
            int64_t const orow = static_cast<int64_t>(t) * outFrame + posOut;
            auto slice = [&](VecIn const(&cin)[NUM_IN], Vec const& cres, Vec& yk, int k)
            {
                VecF const ib = constAt(sInBias, ibReg, k);
#pragma unroll
                for (int i = 0; i < V / 2; ++i)
                {
                    float2 v = pairAt(cin[0], i);
                    if constexpr (IN_WINO)
                    {
                        // sum_q at(J, q) * m_q in index order; the products are exact (powers of
                        // two) so the reference rounding sequence is one rounding per add.
                        constexpr float c0 = W::at(J, termIndex<R>(J, 0));
                        v.x *= c0;
                        v.y *= c0;
#pragma unroll
                        for (int q = 1; q < numTerms<R>(J); ++q)
                        {
                            float const cq = W::at(J, termIndex<R>(J, q)); // constant after unrolling
                            float2 const mq = pairAt(cin[q], i);
                            v.x = addR<T, REF>(v.x, cq * mq.x);
                            v.y = addR<T, REF>(v.y, cq * mq.y);
                        }
                    }
                    if constexpr (HAS_IN_BIAS)
                    {
                        v.x += ib.val[2 * i];
                        v.y += ib.val[2 * i + 1];
                    }
                    // Reference mode (and the row modes) reproduce eager's bf16 roundings of the
                    // conv output and of the residual sum; the slab modes' fp32 mode keeps the fp32
                    // values. The bf16 row is rounded once either way.
                    if constexpr (ROUND1)
                    {
                        v = Elem<T>::toFloat2(Elem<T>::fromFloat2(v));
                    }
                    if constexpr (HAS_RESIDUAL)
                    {
                        float2 const r = pairAt(cres, i);
                        v = make_float2(v.x + r.x, v.y + r.y);
                        if constexpr (ROUND1)
                        {
                            v = Elem<T>::toFloat2(Elem<T>::fromFloat2(v));
                        }
                    }
                    if constexpr (MID != kMidNone)
                    {
                        ss = fmaf(v.x, v.x, ss);
                        ss = fmaf(v.y, v.y, ss);
                    }
                    setPair(yk, i, v);
                }
                if constexpr (MID == kMidNone)
                {
                    storeVec(gRowW(ptrs.y, orow), k, yk); // the combine kernel: the row is the output
                }
                else if (ptrs.sum != nullptr)             // the pre-norm row: the next block's residual
                {
                    storeVec(gRowW(ptrs.sum, static_cast<int64_t>(t) * P + pos), k, yk);
                }
            };
            if constexpr (FRAME_PF)
            {
                slice(inA, resA, y[0], 0); // K == 1: the frame's single slice
            }
            else if constexpr (sliceDepth<K, IN_WINO>() == 1)
            {
                // One slice at a time: loaded right before it is combined (the fewest live
                // registers; latency is hidden by other warps).
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
                    if (k > 0 || !XF_PF)
                    {
                        loadSlice(termsIn, inA, resA, src, k);
                    }
                    slice(inA, resA, y[k], k);
                }
            }
            else if constexpr (sliceDepth<K, IN_WINO>() == 2)
            {
                // Slices in pairs: load the next slice into the other buffer, then compute
                // (with the cross-frame prefetch, slices 0 and 1 are already in).
#pragma unroll
                for (int k = 0; k < K; k += 2)
                {
                    if (k + 1 < K && !(XF_TWO && k == 0))
                    {
                        loadSlice(termsIn, inB, resB, src, k + 1);
                    }
                    slice(inA, resA, y[k], k);
                    if (k + 1 < K)
                    {
                        if (k + 2 < K)
                        {
                            loadSlice(termsIn, inA, resA, src, k + 2);
                        }
                        slice(inB, resB, y[k + 1], k + 1);
                    }
                }
            }
            else
            {
                // Three rotating buffers: two slices in flight while one is normalized.
                if (1 < K)
                {
                    loadSlice(termsIn, inB, resB, src, 1);
                }
#pragma unroll
                for (int k = 0; k < K; k += 3)
                {
                    if (k + 2 < K)
                    {
                        loadSlice(termsIn, inC, resC, src, k + 2);
                    }
                    slice(inA, resA, y[k], k);
                    if (k + 1 < K)
                    {
                        if (k + 3 < K)
                        {
                            loadSlice(termsIn, inA, resA, src, k + 3);
                        }
                        slice(inB, resB, y[k + 1], k + 1);
                    }
                    if (k + 2 < K)
                    {
                        if (k + 4 < K)
                        {
                            loadSlice(termsIn, inB, resB, src, k + 4);
                        }
                        slice(inC, resC, y[k + 2], k + 2);
                    }
                }
            }
            if constexpr (XF_PF)
            {
                if (hasNext)
                {
                    next = srcOf(nextInPos ? t + 1 : 0, nextPos, nextPosSrc);
                    // The next frame's term list is J + 1 within the position, 0 across positions.
                    if (nextInPos)
                    {
                        loadSlice(termsNext, inA, resA, next, 0);
                        if constexpr (XF_TWO)
                        {
                            loadSlice(termsNext, inB, resB, next, 1);
                        }
                    }
                    else
                    {
                        loadSlice(cuda::std::integral_constant<int, 0>{}, inA, resA, next, 0);
                        if constexpr (XF_TWO)
                        {
                            loadSlice(cuda::std::integral_constant<int, 0>{}, inB, resB, next, 1);
                        }
                    }
                }
            }
            if constexpr (MID != kMidNone)
            {
#pragma unroll
                for (int off = L >> 1; off > 0; off >>= 1)
                {
                    ss += __shfl_xor_sync(0xffffffffu, ss, off);
                }
                float denom = 0.0f, invDenom = 0.0f;
                if constexpr (REF)
                {
                    denom = fmaxf(sqrtf(ss), p.eps);
                }
                else
                {
                    invDenom = rsqrtf(fmaxf(ss, p.eps * p.eps));
                }

                // Pass 2: normalize from registers into the ring slot (and emit the slabs this
                // frame completes: slab k with lastFrameOf(k) == J + 2 reads the group's frames
                // f_0 .. f_{J+1} from the ring and f_{J+2}, this frame, from registers) or into
                // the output row.
                int const t0 = t - J; // first output frame of the group; f_i is frame t0 - 2 + i
                int64_t const drow = static_cast<int64_t>(t / R) * outFrame + posOut;
                T* const ySlot = SLABS ? slotPtr(ringSlot<R>(t)) : gRowW(ptrs.y, orow);
#pragma unroll
                for (int k = 0; k < K; ++k)
                {
                    VecF const gm = constAt(sGamma, gReg, k);
                    VecF const bs = constAt(sBias, bReg, k);
                    Vec yk = y[k];
#pragma unroll
                    for (int i = 0; i < V / 2; ++i)
                    {
                        float2 const f = pairAt(yk, i);
                        float2 r;
                        r.x = normElement<T, REF, HAS_BIAS, ACT>(
                            f.x, denom, invDenom, p.scale, gm.val[2 * i], bs.val[2 * i]);
                        r.y = normElement<T, REF, HAS_BIAS, ACT>(
                            f.y, denom, invDenom, p.scale, gm.val[2 * i + 1], bs.val[2 * i + 1]);
                        setPair(yk, i, r);
                    }
                    storeVec(ySlot, k, yk);
                    if constexpr (HALO && !SLABS)
                    {
                        if (haloBaseL != nullptr)
                        {
                            storeVec(
                                gRowW(static_cast<T*>(haloBaseL), static_cast<int64_t>(t) * p.haloFrameL + haloRowL), k,
                                yk);
                        }
                        if (haloBaseR != nullptr)
                        {
                            storeVec(
                                gRowW(static_cast<T*>(haloBaseR), static_cast<int64_t>(t) * p.haloFrameR + haloRowR), k,
                                yk);
                        }
                    }
                    if constexpr (SLABS)
                    {
                        auto emit = [&](auto sTag)
                        {
                            constexpr int s = decltype(sTag)::value;
                            if constexpr (s < W::kComps && lastFrameOf<R>(s) == J + 2)
                            {
                                // Each slab reloads the ring frames it reads (the barrier stops the
                                // compiler from hoisting them across slabs): a few extra LDS keep
                                // the live set small enough not to spill.
                                asm volatile("" ::: "memory");
                                Vec hs[J + 2];
#pragma unroll
                                for (int fi = 0; fi < J + 2; ++fi)
                                {
                                    if (W::bt(s, fi) != 0.0f)
                                    {
                                        hs[fi] = loadVec(slotPtr(ringSlot<R>(t0 - 2 + fi)), k);
                                    }
                                }
                                VecD o;
#pragma unroll
                                for (int i = 0; i < V / 2; ++i)
                                {
                                    // Products of 16-bit values and multiples of 1/2 are exact in
                                    // fp32, so fma and separate multiply-add agree; the slab rounds
                                    // once.
                                    float2 acc = make_float2(0.0f, 0.0f);
                                    bool first = true;
#pragma unroll
                                    for (int fi = 0; fi < J + 3; ++fi)
                                    {
                                        float const coef = W::bt(s, fi); // constant after unrolling
                                        if (coef != 0.0f)
                                        {
                                            float2 const fv
                                                = (fi == J + 2) ? pairAt(yk, i) : pairAt(hs[fi < J + 2 ? fi : 0], i);
                                            if (first)
                                            {
                                                acc = make_float2(coef * fv.x, coef * fv.y);
                                                first = false;
                                            }
                                            else
                                            {
                                                acc.x = fmaf(coef, fv.x, acc.x);
                                                acc.y = fmaf(coef, fv.y, acc.y);
                                            }
                                        }
                                    }
                                    setPair(o, i, acc);
                                }
                                storeVec(gRowW(ptrs.d + s * ptrs.compStride, drow), k, o);
                                if constexpr (HALO)
                                {
                                    if (haloBaseL != nullptr)
                                    {
                                        storeVec(gRowW(static_cast<TD*>(haloBaseL) + s * p.haloCompL,
                                                     static_cast<int64_t>(t / R) * p.haloFrameL + haloRowL),
                                            k, o);
                                    }
                                    if (haloBaseR != nullptr)
                                    {
                                        storeVec(gRowW(static_cast<TD*>(haloBaseR) + s * p.haloCompR,
                                                     static_cast<int64_t>(t / R) * p.haloFrameR + haloRowR),
                                            k, o);
                                    }
                                }
                            }
                        };
                        emit(cuda::std::integral_constant<int, 0>{});
                        emit(cuda::std::integral_constant<int, 1>{});
                        emit(cuda::std::integral_constant<int, 2>{});
                        emit(cuda::std::integral_constant<int, 3>{});
                        emit(cuda::std::integral_constant<int, 4>{});
                        emit(cuda::std::integral_constant<int, 5>{});
                    }
                }
            }
            if constexpr (PF)
            {
                src = next;
                if constexpr (FRAME_PF)
                {
                    if (hasNext)
                    {
#pragma unroll
                        for (int j = 0; j < NUM_IN; ++j)
                        {
                            inA[j] = inB[j];
                        }
                        if constexpr (HAS_RESIDUAL)
                        {
                            resA = resB;
                        }
                    }
                }
                if (!nextInPos)
                {
                    prefetched = hasNext;
                }
            }
            else
            {
                if (nextInPos)
                {
                    src = srcOf(t + 1, pos, posSrc);
                }
            }
        };

        for (int t = 0; t < T_; t += R)
        {
            frame(cuda::std::integral_constant<int, 0>{}, t);
            if constexpr (R >= 2)
            {
                frame(cuda::std::integral_constant<int, 1>{}, t + 1);
            }
            if constexpr (R == 4)
            {
                frame(cuda::std::integral_constant<int, 2>{}, t + 2);
                frame(cuda::std::integral_constant<int, 3>{}, t + 3);
            }
        }
        if constexpr (SLABS)
        {
            // The last two normalized frames are the next chunk's cache.
            T const* const sa = slotPtr(ringSlot<R>(T_ - 2));
            T const* const sb = slotPtr(ringSlot<R>(T_ - 1));
            T* const ta = gRowW(ptrs.yTail, pos);
            T* const tb = gRowW(ptrs.yTail, P + pos);
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
                storeVec(ta, k, loadVec(sa, k));
                storeVec(tb, k, loadVec(sb, k));
            }
        }
    }
}

// Entry point for the runtime-compiled kernel: every argument a scalar (the launcher passes a
// typed argument list).
template <typename T, typename TD, int R, int V, int L, int K, bool REF, bool HAS_BIAS, bool HAS_IN_BIAS,
    bool HAS_RESIDUAL, bool IN_WINO, int MID, int OUT, bool PADDED, bool IN_MAP, bool OUT_MAP, int UP_FT, int UP_RATIO,
    bool HALO>
__global__ void __launch_bounds__(kThreads, blocksPerSm<K, IN_WINO, R, OUT>())
    vaeNormKernel(T const* x, TD const* m, TD* d, long long compStride, long long inCompStride, T const* residual,
        T const* inBias, T const* cache, T const* gamma, T const* bias, T* sum, T* yTail, T* y, long long positions,
        int channels, int frames, int cacheFrames, int height, int width, int padH, int padW, long long inFrame,
        int inW, int inOff, long long outFrame, int outW, int outOff, int upHeight, int upWidth, int upCrop, int upCin,
        float scale, float eps, void* haloL, void* haloR, long long haloFrameL, long long haloFrameR,
        long long haloCompL, long long haloCompR, int haloStepL, int haloStepR, int haloAxis)
{
    Ptrs<T, TD> ptrs{};
    ptrs.x = x;
    ptrs.m = m;
    ptrs.d = d;
    ptrs.compStride = compStride;
    ptrs.inCompStride = inCompStride;
    ptrs.residual = residual;
    ptrs.inBias = inBias;
    ptrs.cache = cache;
    ptrs.gamma = gamma;
    ptrs.bias = bias;
    ptrs.sum = sum;
    ptrs.yTail = yTail;
    ptrs.y = y;
    VaeNormParams p{};
    p.positions = positions;
    p.channels = channels;
    p.frames = frames;
    p.cacheFrames = cacheFrames;
    p.height = height;
    p.width = width;
    p.padH = padH;
    p.padW = padW;
    p.inFrame = inFrame;
    p.inW = inW;
    p.inOff = inOff;
    p.outFrame = outFrame;
    p.outW = outW;
    p.outOff = outOff;
    p.upHeight = upHeight;
    p.upWidth = upWidth;
    p.upCrop = upCrop;
    p.upCin = upCin;
    p.scale = scale;
    p.eps = eps;
    p.haloL = haloL;
    p.haloR = haloR;
    p.haloFrameL = haloFrameL;
    p.haloFrameR = haloFrameR;
    p.haloCompL = haloCompL;
    p.haloCompR = haloCompR;
    p.haloStepL = haloStepL;
    p.haloStepR = haloStepR;
    p.haloAxis = haloAxis;
    vaeNormBody<T, TD, R, V, L, K, REF, HAS_BIAS, HAS_IN_BIAS, HAS_RESIDUAL, IN_WINO, MID, OUT, PADDED, IN_MAP, OUT_MAP,
        UP_FT, UP_RATIO, HALO>(ptrs, p);
}

} // namespace vaewino
