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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/gdnVerifyKernels.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// One CTA per (request, value head): 4 warps, warp w owns value columns [32 w, 32 w + 32). The head-dim products run
// on m16n8k16 BF16 MMAs whose A operand rows are value columns (or tokens) and whose B columns are tokens (or
// history rows); the gated RMSNorm of a head reduces inside the CTA.
constexpr int kDim = kGdnVerifyHeadDim;
constexpr int kHist = kGdnVerifyHistory;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr unsigned kFullMask = 0xffffffffu;
// Query scale 1 / sqrt(128).
constexpr float kQScale = 0.08838834764831845F;

static_assert(kDim == 128 && kHist == 16 && kGdnVerifyMaxTokens == 8, "the lane layout below assumes these shapes");

__device__ __forceinline__ float bfLo(uint32_t x)
{
    return __int_as_float(x << 16);
}

__device__ __forceinline__ float bfHi(uint32_t x)
{
    return __int_as_float(x & 0xffff0000u);
}

__device__ __forceinline__ float bfToFloat(uint16_t x)
{
    return __int_as_float(static_cast<uint32_t>(x) << 16);
}

__device__ __forceinline__ uint16_t floatToBf(float x)
{
    uint16_t r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float bfRound(float x)
{
    return bfToFloat(floatToBf(x));
}

// (lo, hi) -> bf16x2 with lo in the low half.
__device__ __forceinline__ uint32_t packBf2(float lo, float hi)
{
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ uint32_t word(uint4 const& v, int i)
{
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint4 load16(void const* p)
{
    return *static_cast<uint4 const*>(p);
}

__device__ __forceinline__ void store16(void* p, uint4 v)
{
    *static_cast<uint4*>(p) = v;
}

__device__ __forceinline__ uint4 zero4()
{
    return make_uint4(0u, 0u, 0u, 0u);
}

__device__ __forceinline__ float sqrtApprox(float x)
{
    float r;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float rcpApprox(float x)
{
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float rsqrtApprox(float x)
{
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float rcpRn(float x)
{
    float r;
    asm("div.rn.f32 %0, 0f3F800000, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ uint8_t floatToE4m3(float x)
{
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(r) : "f"(x));
    return static_cast<uint8_t>(r & 0xff);
}

__device__ __forceinline__ float pick8(float const (&a)[8], int i)
{
    float r = a[0];
#pragma unroll
    for (int k = 1; k < 8; ++k)
    {
        r = (i == k) ? a[k] : r;
    }
    return r;
}

__device__ __forceinline__ void mma16816(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// K-reduction over 128-wide rows: A = rows r (rows r + 8 are zero), B = 8 columns (rows of B). The lane (r, q) holds
// the physical elements [32 c + 8 q, 32 c + 8 q + 8) of its row in chunk c; k-step 2c + e uses words (2e, 2e + 1)
// for both operands.
__device__ __forceinline__ void rowDot(float (&acc)[4], uint4 const (&a)[4], uint4 const (&b)[4])
{
#pragma unroll
    for (int c = 0; c < 4; ++c)
    {
#pragma unroll
        for (int e = 0; e < 2; ++e)
        {
            mma16816(acc, word(a[c], 2 * e), 0u, word(a[c], 2 * e + 1), 0u, word(b[c], 2 * e), word(b[c], 2 * e + 1));
        }
    }
}

// PDL: before griddepcontrol.wait the kernel reads the layer weights and the data of earlier steps (slots, accepted
// counts, history, replay state), none of which its producer (in_proj / conv1d) writes; this step's q / k / v / z / a /
// b rows are read after it. Dependents are released once every input is read. Pointers to data that other launches
// write are not __restrict__, so their loads are never ld.global.nc, which ptxas may move ahead of the wait.
template <int kTokens, bool kOutFp8>
__global__ void __launch_bounds__(kThreads, 1) gdnVerifyKernel(uint16_t const* x, int64_t xStride, uint16_t const* z,
    int64_t zStride, uint16_t const* aIn, uint16_t const* bIn, int64_t abStride, float const* __restrict__ logA,
    float const* __restrict__ dtBias, uint16_t const* __restrict__ normWeight, float const* __restrict__ fp8Scale,
    float eps, float qScale, uint16_t* state, int64_t stateSlotStride, int32_t const* stateIndices, uint16_t* oldU,
    uint16_t* oldK, float* oldG, int32_t const* cacheBufIdx, int32_t const* prevNumAccepted, void* y, int64_t yStride,
    int numKHeads, int numVHeads)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    constexpr int kT = kTokens;
    int const hk_ = numKHeads;
    int const hv_ = numVHeads;
    int const group = hv_ / hk_; // value heads per key head
    int const n = blockIdx.x / hv_;
    int const hv = blockIdx.x - n * hv_;
    int const hk = hv / group;
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const r = lane >> 2;
    int const q = lane & 3;
    int const vb = warp * 32; // value columns vb .. vb + 31: m-tile m, half h -> v = vb + 16 m + 8 h + r
    int const t0 = 2 * q;
    int const t1 = 2 * q + 1;
    int64_t const row0 = static_cast<int64_t>(n) * kT;
    int64_t const kStrideK = static_cast<int64_t>(hk_) * kDim; // history key row
    int64_t const kStrideU = static_cast<int64_t>(hv_) * kDim; // history value row

    __shared__ float2 sCoef[2][2][32]; // [k / q][j-tile][lane]: raw history product x decay (scaled by 1/|row| on use)
    __shared__ float sInv[2][8];       // 1 / |k_t|, scale / |q_t|
    __shared__ float sA[8][8];         // A[t][s]
    __shared__ float sQ[8][8];         // raw q_t k_s x decay, t >= s (scaled by inv_q[t] inv_k[s] on use)
    __shared__ float sNorm[kWarps][8];

    float const negAExp = -__expf(logA[hv]);
    float const dtb = dtBias[hv];
    float nw[2][2];
#pragma unroll
    for (int m = 0; m < 2; ++m)
    {
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            nw[m][h] = bfToFloat(normWeight[vb + 16 * m + 8 * h + r]);
        }
    }
    float rScale = 1.f;
    if constexpr (kOutFp8)
    {
        rScale = rcpRn(fp8Scale[0]);
    }

    // This step's in_proj output rows (read after the dependency wait below).
    int const slot = stateIndices[n];
    bool const ok0 = t0 < kT;
    bool const ok1 = t1 < kT;
    uint16_t const* xq = x + (row0 + (r < kT ? r : 0)) * xStride + hk * kDim + 8 * q;
    uint16_t const* xk = xq + kStrideK;
    uint16_t const* xv0 = x + (row0 + t0) * xStride + 2 * kStrideK + hv * kDim + vb + r;
    uint16_t const* xv1 = xv0 + xStride;
    uint16_t const* zv0 = z + (row0 + t0) * zStride + hv * kDim + vb + r;
    uint16_t const* zv1 = zv0 + zStride;
    uint16_t const* pa = aIn + (row0 + (lane < kT ? lane : 0)) * abStride + hv;
    uint16_t const* pb = bIn + (row0 + (lane < kT ? lane : 0)) * abStride + hv;

    // Data of earlier steps (slot, accepted count, history, state), which the producer does not write: read
    // before the dependency wait so the loads overlap it (a padded slot reads slot 0 and returns below).
    int const slotR = slot < 0 ? 0 : slot;
    int const pn = prevNumAccepted[slotR];
    int const buf = cacheBufIdx[slotR];
    bool const isWrite = pn + kT > kHist;
    int const wBuf = isWrite ? 1 - buf : buf;
    int const wOff = isWrite ? 0 : pn;
    int64_t const rb = static_cast<int64_t>(slotR) * 2 + buf;
    int64_t const wb = static_cast<int64_t>(slotR) * 2 + wBuf;
    uint16_t const* khp = oldK + rb * (kHist * kStrideK) + hk * kDim;
    uint16_t const* uhp = oldU + rb * (kHist * kStrideU) + hv * kDim;
    float const* ghp = oldG + (rb * hv_ + hv) * kHist;
    uint16_t* sp = state + slotR * stateSlotStride + static_cast<int64_t>(hv) * kDim * kDim;

    uint4 H[2][2][4]; // checkpoint state rows (h^T) of this lane: [m-tile][half][chunk]
#pragma unroll
    for (int m = 0; m < 2; ++m)
    {
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                H[m][h][c] = load16(sp + (vb + 16 * m + 8 * h + r) * kDim + 32 * c + 8 * q);
            }
        }
    }
    // History columns of this lane: j = 2q, 2q + 1, 2q + 8, 2q + 9 (rows j >= pn are masked to zero).
    int const jj[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
    bool hj[4];
    float gh[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        hj[i] = jj[i] < pn;
        gh[i] = hj[i] ? ghp[jj[i]] : 0.f;
    }
    float const gStart = pn > 0 ? ghp[pn - 1] : 0.f;
    // uh^T as the A operand of the history outputs:
    // [m-tile][a0..a3] = (v0, j 2q..2q+1), (v1, ..), (v0, j + 8), (v1, j + 8)
    uint32_t UA[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
    {
        int const va = vb + 16 * m + r;
        int const vc = va + 8;
        uint16_t e[8];
        e[0] = hj[0] ? uhp[jj[0] * kStrideU + va] : uint16_t{0};
        e[1] = hj[1] ? uhp[jj[1] * kStrideU + va] : uint16_t{0};
        e[2] = hj[0] ? uhp[jj[0] * kStrideU + vc] : uint16_t{0};
        e[3] = hj[1] ? uhp[jj[1] * kStrideU + vc] : uint16_t{0};
        e[4] = hj[2] ? uhp[jj[2] * kStrideU + va] : uint16_t{0};
        e[5] = hj[3] ? uhp[jj[3] * kStrideU + va] : uint16_t{0};
        e[6] = hj[2] ? uhp[jj[2] * kStrideU + vc] : uint16_t{0};
        e[7] = hj[3] ? uhp[jj[3] * kStrideU + vc] : uint16_t{0};
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            UA[m][i] = static_cast<uint32_t>(e[2 * i]) | (static_cast<uint32_t>(e[2 * i + 1]) << 16);
        }
    }
    // History keys (warps 1 and 2: the history products), rows j = r + 8 jt, rows >= pn zero.
    uint4 KH[2][4];
#pragma unroll
    for (int jt = 0; jt < 2; ++jt)
    {
        int const j = r + 8 * jt;
        bool const ok = (warp == 1 || warp == 2) && j < pn;
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            KH[jt][c] = ok ? load16(khp + j * kStrideK + 32 * c + 8 * q) : zero4();
        }
    }
    cudaGridDependencySynchronize();

    // Raw q / k rows (row r = token; rows >= T zero), v / z of this lane's (v, t), a / b of token = lane.
    uint4 QR[4], KR[4];
    bool const rowOk = r < kT;
#pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        QR[c] = rowOk ? load16(xq + 32 * c) : zero4();
        KR[c] = rowOk ? load16(xk + 32 * c) : zero4();
    }
    float vv[2][4], zz[2][4]; // [m][(h, t)]: 0 (h0, t0), 1 (h0, t1), 2 (h1, t0), 3 (h1, t1)
#pragma unroll
    for (int m = 0; m < 2; ++m)
    {
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int const off = 16 * m + 8 * h;
            vv[m][2 * h] = ok0 ? bfToFloat(xv0[off]) : 0.f;
            vv[m][2 * h + 1] = ok1 ? bfToFloat(xv1[off]) : 0.f;
            zz[m][2 * h] = ok0 ? bfToFloat(zv0[off]) : 0.f;
            zz[m][2 * h + 1] = ok1 ? bfToFloat(zv1[off]) : 0.f;
        }
    }
    float ga = 0.f;
    float gb = 0.f;
    if (lane < kT)
    {
        ga = bfToFloat(*pa);
        gb = bfToFloat(*pb);
    }

    if (slot < 0)
    {
        // Padded request: defined zeros for out_proj.
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int const t = (i & 1) ? t1 : t0;
            int const vs = vb + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + r;
            if (t < kT)
            {
                int64_t const idx = (row0 + t) * yStride + hv * kDim + vs;
                if constexpr (kOutFp8)
                {
                    static_cast<uint8_t*>(y)[idx] = 0;
                }
                else
                {
                    static_cast<uint16_t*>(y)[idx] = 0;
                }
            }
        }
        return;
    }

    // Checkpoint commit (history full): h' = h e^{gStart} + kh^T Uc, Uc[j, v] = u[j, v] e^{gStart - G_j} (j < pn),
    // Uc split into bf16 hi + lo. Transposed: h'^T[v, kd] = h^T e^{gStart} + Uc^T[v, j] kh[j, kd]; the n-tile (c, wd)
    // covers the physical kd of word wd of chunk c of this lane's state rows, which only this lane reads and writes.
    if (isWrite)
    {
        float const eg = __expf(gStart);
        float cd[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            cd[i] = hj[i] ? __expf(gStart - gh[i]) : 0.f;
        }
        uint32_t AH[2][4], AL[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                // a_i: (half i & 1, j pair i >> 1): elements j = jj[2 (i >> 1)], jj[2 (i >> 1) + 1]
                float const lo = bfLo(UA[m][i]) * cd[2 * (i >> 1)];
                float const hi = bfHi(UA[m][i]) * cd[2 * (i >> 1) + 1];
                AH[m][i] = packBf2(lo, hi);
                AL[m][i] = packBf2(lo - bfLo(AH[m][i]), hi - bfHi(AH[m][i]));
            }
        }
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            uint32_t out[2][2][4];
#pragma unroll
            for (int wd = 0; wd < 4; ++wd)
            {
                // B column n = r holds kd = 32c + 8 (r >> 1) + 2 wd + (r & 1)
                int const kd = 32 * c + 8 * (r >> 1) + 2 * wd + (r & 1);
                uint16_t const k0 = hj[0] ? khp[jj[0] * kStrideK + kd] : uint16_t{0};
                uint16_t const k1 = hj[1] ? khp[jj[1] * kStrideK + kd] : uint16_t{0};
                uint16_t const k2 = hj[2] ? khp[jj[2] * kStrideK + kd] : uint16_t{0};
                uint16_t const k3 = hj[3] ? khp[jj[3] * kStrideK + kd] : uint16_t{0};
                uint32_t const b0 = static_cast<uint32_t>(k0) | (static_cast<uint32_t>(k1) << 16);
                uint32_t const b1 = static_cast<uint32_t>(k2) | (static_cast<uint32_t>(k3) << 16);
#pragma unroll
                for (int m = 0; m < 2; ++m)
                {
                    float acc[4] = {0.f, 0.f, 0.f, 0.f};
                    mma16816(acc, AH[m][0], AH[m][1], AH[m][2], AH[m][3], b0, b1);
                    mma16816(acc, AL[m][0], AL[m][1], AL[m][2], AL[m][3], b0, b1);
                    uint32_t const hw0 = word(H[m][0][c], wd);
                    uint32_t const hw1 = word(H[m][1][c], wd);
                    out[m][0][wd] = packBf2(bfLo(hw0) * eg + acc[0], bfHi(hw0) * eg + acc[1]);
                    out[m][1][wd] = packBf2(bfLo(hw1) * eg + acc[2], bfHi(hw1) * eg + acc[3]);
                }
            }
#pragma unroll
            for (int m = 0; m < 2; ++m)
            {
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    store16(sp + (vb + 16 * m + 8 * h + r) * kDim + 32 * c + 8 * q,
                        make_uint4(out[m][h][0], out[m][h][1], out[m][h][2], out[m][h][3]));
                }
            }
        }
    }

    // Gating (lane t < T), cumulative log-decay by an 8-lane scan.
    float gl = 0.f;
    float beta = 0.f;
    if (lane < kT)
    {
        float const xg = ga + dtb;
        float const softplus = xg <= 20.f ? 0.6931471805599453f * __log2f(1.f + __expf(xg)) : xg;
        gl = negAExp * softplus;
        beta = __fdividef(1.f, 1.f + __expf(-gb));
    }
#pragma unroll
    for (int d = 1; d < 8; d <<= 1)
    {
        float const o = __shfl_up_sync(kFullMask, gl, d);
        if (lane >= d)
        {
            gl += o;
        }
    }
    float const gv = gStart + gl; // lane t: G[t]
    float const gt0 = __shfl_sync(kFullMask, gv, t0);
    float const gt1 = __shfl_sync(kFullMask, gv, t1);
    float const bt0 = __shfl_sync(kFullMask, beta, t0);
    float const bt1 = __shfl_sync(kFullMask, beta, t1);
    float const gr = __shfl_sync(kFullMask, gv, r);
    float const br = __shfl_sync(kFullMask, beta, r);
    float const eG0 = __expf(gt0);
    float const eG1 = __expf(gt1);

    // State products on the raw rows (all warps): rows v, columns = tokens; scaled by 1 / |k_t|, scale / |q_t| later.
    float KS[2][4], QS[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            KS[m][i] = 0.f;
            QS[m][i] = 0.f;
        }
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
#pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                uint32_t const a0 = word(H[m][0][c], 2 * e);
                uint32_t const a1 = word(H[m][1][c], 2 * e);
                uint32_t const a2 = word(H[m][0][c], 2 * e + 1);
                uint32_t const a3 = word(H[m][1][c], 2 * e + 1);
                mma16816(KS[m], a0, a1, a2, a3, word(KR[c], 2 * e), word(KR[c], 2 * e + 1));
                mma16816(QS[m], a0, a1, a2, a3, word(QR[c], 2 * e), word(QR[c], 2 * e + 1));
            }
        }
    }

    // Head-common products: warp 3: k k^T, q q^T (norms, A); warp 0: q k^T; warps 1 / 2: k / q x history keys.
    if (warp == 3)
    {
        float kk[4] = {0.f, 0.f, 0.f, 0.f};
        float qq[4] = {0.f, 0.f, 0.f, 0.f};
        rowDot(kk, KR, KR);
        rowDot(qq, QR, QR);
        // |k_t|^2 = (k k^T)[t, t] sits in lane 4t + (t >> 1), element t & 1.
        float const dk = (r & 1) ? kk[1] : kk[0];
        float const dq = (r & 1) ? qq[1] : qq[0];
        float const ik = rcpApprox(sqrtApprox(dk + 1e-6f));
        float const iq = qScale * rcpApprox(sqrtApprox(dq + 1e-6f));
        float const ikR = __shfl_sync(kFullMask, ik, 4 * r + (r >> 1));
        float const ik0 = __shfl_sync(kFullMask, ik, 4 * t0 + q);
        float const ik1 = __shfl_sync(kFullMask, ik, 4 * t1 + q);
        if (q == (r >> 1))
        {
            sInv[0][r] = ik;
            sInv[1][r] = iq;
        }
        sA[r][t0] = (r > t0 && r < kT) ? br * (kk[0] * ikR * ik0) * __expf(gr - gt0) : 0.f;
        sA[r][t1] = (r > t1 && r < kT) ? br * (kk[1] * ikR * ik1) * __expf(gr - gt1) : 0.f;
    }
    else if (warp == 0)
    {
        float qk[4] = {0.f, 0.f, 0.f, 0.f};
        rowDot(qk, QR, KR);
        sQ[r][t0] = (r >= t0 && r < kT) ? qk[0] * __expf(gr - gt0) : 0.f;
        sQ[r][t1] = (r >= t1 && r < kT) ? qk[1] * __expf(gr - gt1) : 0.f;
    }
    else
    {
        // (t = r, j = 8 jt + 2q + i): raw product x e^{G_t - G_j}, j < pn
#pragma unroll
        for (int jt = 0; jt < 2; ++jt)
        {
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            if (warp == 1)
            {
                rowDot(acc, KR, KH[jt]);
            }
            else
            {
                rowDot(acc, QR, KH[jt]);
            }
            float2 cf;
            cf.x = hj[2 * jt] ? acc[0] * __expf(gr - gh[2 * jt]) : 0.f;
            cf.y = hj[2 * jt + 1] ? acc[1] * __expf(gr - gh[2 * jt + 1]) : 0.f;
            sCoef[warp - 1][jt][lane] = cf;
        }
    }
    __syncthreads(); // every global input of the CTA is read
    cudaTriggerProgrammaticLaunchCompletion();

    float const ik0 = sInv[0][t0];
    float const ik1 = sInv[0][t1];
    float const iq0 = sInv[1][t0];
    float const iq1 = sInv[1][t1];
    float const ikr = sInv[0][r];
    float const iqr = sInv[1][r];
    // History of this step's tokens: normalized keys (one value head per key head, warp 0) and G (warp 1).
    if (warp == 0 && (hv % group) == 0 && r < kT)
    {
        uint16_t* wk = oldK + wb * (kHist * kStrideK) + (wOff + r) * kStrideK + hk * kDim + 8 * q;
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            uint4 v4;
            v4.x = packBf2(bfLo(KR[c].x) * ikr, bfHi(KR[c].x) * ikr);
            v4.y = packBf2(bfLo(KR[c].y) * ikr, bfHi(KR[c].y) * ikr);
            v4.z = packBf2(bfLo(KR[c].z) * ikr, bfHi(KR[c].z) * ikr);
            v4.w = packBf2(bfLo(KR[c].w) * ikr, bfHi(KR[c].w) * ikr);
            store16(wk + 32 * c, v4);
        }
    }
    if (warp == 1 && lane < kT)
    {
        oldG[(wb * hv_ + hv) * kHist + wOff + lane] = isWrite ? gl : gv;
    }

    // History outputs: uh^T (16 x 16) x coeff^T (16 x 8), coefficient = raw x decay x (1 / |row t|) in bf16.
    uint32_t bk0, bk1, bq0, bq1;
    {
        float2 const ck0 = sCoef[0][0][lane];
        float2 const ck1 = sCoef[0][1][lane];
        float2 const cq0 = sCoef[1][0][lane];
        float2 const cq1 = sCoef[1][1][lane];
        bk0 = packBf2(ck0.x * ikr, ck0.y * ikr);
        bk1 = packBf2(ck1.x * ikr, ck1.y * ikr);
        bq0 = packBf2(cq0.x * iqr, cq0.y * iqr);
        bq1 = packBf2(cq1.x * iqr, cq1.y * iqr);
    }
    float KHI[2][4], QHI[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            KHI[m][i] = 0.f;
            QHI[m][i] = 0.f;
        }
        mma16816(KHI[m], UA[m][0], UA[m][1], UA[m][2], UA[m][3], bk0, bk1);
        mma16816(QHI[m], UA[m][0], UA[m][1], UA[m][2], UA[m][3], bq0, bq1);
    }
    float const eGk0 = eG0 * ik0;
    float const eGk1 = eG1 * ik1;
    float const eGq0 = eG0 * iq0;
    float const eGq1 = eG1 * iq1;
    // rhs = beta (v - e^G kh0 - k_hist); C layout per m-tile: [0] (h0, t0), [1] (h0, t1), [2] (h1, t0), [3] (h1, t1)
    float rhs[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
    {
        rhs[m][0] = bt0 * (vv[m][0] - eGk0 * KS[m][0] - KHI[m][0]);
        rhs[m][1] = bt1 * (vv[m][1] - eGk1 * KS[m][1] - KHI[m][1]);
        rhs[m][2] = bt0 * (vv[m][2] - eGk0 * KS[m][2] - KHI[m][2]);
        rhs[m][3] = bt1 * (vv[m][3] - eGk1 * KS[m][3] - KHI[m][3]);
    }

    // Forward substitution (I + A) U = rhs per value column; every lane of a quad solves its four columns.
    float U[4][8];
    {
        float R[4][8];
#pragma unroll
        for (int qq = 0; qq < 4; ++qq)
        {
            int const src = (lane & ~3) | qq;
#pragma unroll
            for (int m = 0; m < 2; ++m)
            {
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    R[2 * m + h][2 * qq] = __shfl_sync(kFullMask, rhs[m][2 * h], src);
                    R[2 * m + h][2 * qq + 1] = __shfl_sync(kFullMask, rhs[m][2 * h + 1], src);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            float u[4] = {0.f, 0.f, 0.f, 0.f};
            if (i < kT)
            {
#pragma unroll
                for (int col = 0; col < 4; ++col)
                {
                    u[col] = R[col][i];
                }
#pragma unroll
                for (int s = 0; s < i; ++s)
                {
                    float const a = sA[i][s];
#pragma unroll
                    for (int col = 0; col < 4; ++col)
                    {
                        u[col] -= a * U[col][s];
                    }
                }
            }
#pragma unroll
            for (int col = 0; col < 4; ++col)
            {
                U[col][i] = u[col];
            }
        }
    }
    // Intra-chunk output of this lane's tokens: q_new[v, t] = sum_{s <= t} inv_q[t] inv_k[s] Qraw[t][s] U[v][s].
    float qn[2][4];
    {
        float c0[8], c1[8];
#pragma unroll
        for (int s = 0; s < 8; ++s)
        {
            float const iks = s < kT ? sInv[0][s] : 0.f;
            c0[s] = s < kT ? sQ[t0][s] * iks * iq0 : 0.f;
            c1[s] = s < kT ? sQ[t1][s] * iks * iq1 : 0.f;
        }
#pragma unroll
        for (int m = 0; m < 2; ++m)
        {
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                float a0 = 0.f;
                float a1 = 0.f;
#pragma unroll
                for (int s = 0; s < kT; ++s)
                {
                    a0 += c0[s] * U[2 * m + h][s];
                    a1 += c1[s] * U[2 * m + h][s];
                }
                qn[m][2 * h] = a0;
                qn[m][2 * h + 1] = a1;
            }
        }
    }
    float ov[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
    {
        ov[m][0] = eGq0 * QS[m][0] + QHI[m][0] + qn[m][0];
        ov[m][1] = eGq1 * QS[m][1] + QHI[m][1] + qn[m][1];
        ov[m][2] = eGq0 * QS[m][2] + QHI[m][2] + qn[m][2];
        ov[m][3] = eGq1 * QS[m][3] + QHI[m][3] + qn[m][3];
    }

    // History of this step's tokens: U.
    {
        uint16_t* wu = oldU + wb * (kHist * kStrideU) + hv * kDim + vb + r;
#pragma unroll
        for (int col = 0; col < 4; ++col)
        {
            int const off = 16 * (col >> 1) + 8 * (col & 1);
            if (ok0)
            {
                wu[(wOff + t0) * kStrideU + off] = floatToBf(pick8(U[col], t0));
            }
            if (ok1)
            {
                wu[(wOff + t1) * kStrideU + off] = floatToBf(pick8(U[col], t1));
            }
        }
    }

    // Gated RMSNorm over the 128 columns of the head, on the bf16-rounded replay output.
    float xb[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            xb[m][i] = bfRound(ov[m][i]);
        }
    }
    float p0 = xb[0][0] * xb[0][0] + xb[0][2] * xb[0][2] + xb[1][0] * xb[1][0] + xb[1][2] * xb[1][2];
    float p1 = xb[0][1] * xb[0][1] + xb[0][3] * xb[0][3] + xb[1][1] * xb[1][1] + xb[1][3] * xb[1][3];
#pragma unroll
    for (int msk = 4; msk < 32; msk <<= 1)
    {
        p0 += __shfl_xor_sync(kFullMask, p0, msk);
        p1 += __shfl_xor_sync(kFullMask, p1, msk);
    }
    if (r == 0)
    {
        sNorm[warp][t0] = p0;
        sNorm[warp][t1] = p1;
    }
    __syncthreads();
    float ss0 = 0.f;
    float ss1 = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
    {
        ss0 += sNorm[w][t0];
        ss1 += sNorm[w][t1];
    }
    float const rstd0 = rsqrtApprox(ss0 * (1.f / kDim) + eps);
    float const rstd1 = rsqrtApprox(ss1 * (1.f / kDim) + eps);
#pragma unroll
    for (int m = 0; m < 2; ++m)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int const h = i >> 1;
            bool const ok = (i & 1) ? ok1 : ok0;
            float yv = xb[m][i] * ((i & 1) ? rstd1 : rstd0) * nw[m][h];
            float const zv = zz[m][i];
            yv *= zv * __fdividef(1.f, 1.f + __expf(-zv));
            if (ok)
            {
                int64_t const idx = (row0 + ((i & 1) ? t1 : t0)) * yStride + hv * kDim + vb + r + 16 * m + 8 * h;
                if constexpr (kOutFp8)
                {
                    static_cast<uint8_t*>(y)[idx] = floatToE4m3(bfRound(yv) * rScale);
                }
                else
                {
                    static_cast<uint16_t*>(y)[idx] = floatToBf(yv);
                }
            }
        }
    }
#endif
}

template <int kTokens, bool kOutFp8>
void launchGdnVerify(GdnVerifyParams const& p, cudaStream_t stream)
{
    dim3 const grid(p.numRequests * p.numVHeads);
    dim3 const block(kThreads);
    tensorrt_llm::common::launchWithPdlWhenEnabled("gdnVerify", gdnVerifyKernel<kTokens, kOutFp8>, grid, block, 0,
        stream, static_cast<uint16_t const*>(p.qkv), p.qkvStride, static_cast<uint16_t const*>(p.z), p.zStride,
        static_cast<uint16_t const*>(p.a), static_cast<uint16_t const*>(p.b), p.abStride, p.logA, p.dtBias,
        static_cast<uint16_t const*>(p.normWeight), p.fp8Scale, p.normEps, kQScale, static_cast<uint16_t*>(p.ssmStates),
        p.ssmSlotStride, p.stateIndices, static_cast<uint16_t*>(p.oldU), static_cast<uint16_t*>(p.oldK), p.oldG,
        p.cacheBufIdx, p.prevNumAcceptedTokens, p.out, p.outStride, p.numKHeads, p.numVHeads);
}

template <bool kOutFp8>
void dispatchTokens(GdnVerifyParams const& p, cudaStream_t stream)
{
    switch (p.numTokens)
    {
    case 2: launchGdnVerify<2, kOutFp8>(p, stream); break;
    case 3: launchGdnVerify<3, kOutFp8>(p, stream); break;
    case 4: launchGdnVerify<4, kOutFp8>(p, stream); break;
    case 5: launchGdnVerify<5, kOutFp8>(p, stream); break;
    case 6: launchGdnVerify<6, kOutFp8>(p, stream); break;
    case 7: launchGdnVerify<7, kOutFp8>(p, stream); break;
    case 8: launchGdnVerify<8, kOutFp8>(p, stream); break;
    default: TLLM_THROW("gdnVerify: unsupported draft step width %d", p.numTokens);
    }
}

} // namespace

void invokeGdnVerify(GdnVerifyParams const& params, cudaStream_t stream)
{
    if (params.fp8Scale != nullptr)
    {
        dispatchTokens<true>(params, stream);
    }
    else
    {
        dispatchTokens<false>(params, stream);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
