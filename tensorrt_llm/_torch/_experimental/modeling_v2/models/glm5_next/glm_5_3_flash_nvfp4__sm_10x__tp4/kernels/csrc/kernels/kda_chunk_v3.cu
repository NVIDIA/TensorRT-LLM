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
// Chunked KDA prefill, v3: BF16 tensor-core prep and a fused state / output scan.
//
// Same math as kda_chunk.cu (chunk size 64, per head S the [K, V] fp32 state, G the in-chunk
// cumulative log decay per channel, every exponent a decay difference):
//   L[t,s] = k_t . k_s e^{G_t - G_s} (s < t),  M[t,s] = q_t . k_s e^{G_t - G_s} / sqrt(D) (s <= t)
//   AB = (I + diag(b) L)^-1 diag(b),  W = AB (k e^G),  U = AB V - W S,  O = q e^G S / sqrt(D) + M U
//   S <- e^{G_C} S + Kd^T U,  Kd = k e^{G_C - G}
// prep (grid (num_chunks, H), 4 warps, three CTAs per SM): a decay pass (thread = channel pair x
//   row half; the gates and G in registers) writes bf16 operand tiles whose decay is relative to the
//   middle row m of each 16-row block (q, k x e^{G - G_m}; k x e^{G_m - G}), and Qh, Kd; L and M are
//   ldmatrix + BF16 mma.sync tiles (the B side of an off-diagonal tile times e^{G_m(a) - G_m(b)} <= 1
//   per channel, so no factor leaves e^{+-8 x lower bound}); T^-1 in fp32 (diagonal 16 x 16 blocks
//   by forward substitution in registers, the off-diagonal ones by TF32 mma.sync, each warp walking
//   down one column block); W = AB (k e^G) on BF16 mma.sync. The output is an image of the scan's
//   shared-memory tiles: e^{G_C} | the chunk's extent | -W | Qh | Kd | M | AB (padded rows).
// scan (grid (B, H, 2 or 4)): a pair of warps owns 16 state columns (columns evolve independently),
//   one warp per half of the state rows / of the chunk rows; per chunk the pair exchanges its bf16
//   state (published by the chunk's CTA barrier) and U (pair barrier) through shared memory. The tile
//   image arrives with one bulk async copy per chunk (two-stage mbarrier ring), V by cp.async.
//   32-column CTAs (four per head, one warp per scheduler) when the grid leaves SMs idle, else 64.
#include "pdl.cuh"
#include "kda.h"
#include "reduce.cuh"

namespace glm {
namespace k {

namespace v3 {
constexpr int kC = 64;    // chunk length
constexpr int kD = 128;   // head dim
constexpr int kHP = kD + 8;       // 16-bit rows of 128 (conflict-free ldmatrix)
constexpr int kAP = kC + 8;       // 16-bit rows of 64
// tile image per (chunk, head), bytes: e^{G_C} fp32 [128] | the chunk's (first token, length) int2 +
// pad | -W [64][kHP] | Qh [64][kHP] | Kd [64][kHP] | M [64][kAP] | AB [64][kAP] (bf16; the padding
// columns are never read)
constexpr int kImgCh = kD * 4;
constexpr int kImgW = kImgCh + 16;
constexpr int kImgQh = kImgW + kC * kHP * 2;
constexpr int kImgKd = kImgQh + kC * kHP * 2;
constexpr int kImgM = kImgKd + kC * kHP * 2;
constexpr int kImgAB = kImgM + kC * kAP * 2;
constexpr int kImgBytes = kImgAB + kC * kAP * 2;
constexpr size_t kBlk = kImgBytes / 4;           // floats per block
static_assert(kImgBytes % 16 == 0, "bulk copy granularity");

constexpr int kPT = 128;          // prep threads
constexpr int kGP = kD + 4;       // fp32 G rows
constexpr int kXP = kC + 1;       // fp32 L / T^-1 rows

struct PrepSmem {
  float G[kC][kGP];     // cumulative log decay; later the off-diagonal inverse's scratch
  __half q[kC][kHP];    // raw q (bf16) -> l2-normalized q (fp16) -> k e^G (bf16, B operand of W)
  __half k[kC][kHP];    // raw k (bf16) -> l2-normalized k (fp16) -> AB (bf16 [64][kAP])
  float X[kC][kXP];     // raw g (bf16 [64][128]) -> L^T (strict upper) / T^-1 (lower)
  float beta[kC];
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// (not volatile: the compiler may hoist fragment loads ahead of independent MMAs; the memory
// clobber keeps the loads ordered with shared-memory stores and barriers)
__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(smem_u32(p))
      : "memory");
}
__device__ __forceinline__ void ldsm4t(uint32_t* r, const void* p) {
  asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(smem_u32(p))
      : "memory");
}
__device__ __forceinline__ void ldsm2t(uint32_t* r, const void* p) {
  asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(r[0]), "=r"(r[1])
      : "r"(smem_u32(p))
      : "memory");
}
// B fragments (k 16 x n 8 NI) of a row-major [k][n] tile: one (x2) or two (x4) n-tiles
template <int NI>
__device__ __forceinline__ void ldsm_bt(uint32_t* r, const void* p) {
  if constexpr (NI == 2)
    ldsm4t(r, p);
  else
    ldsm2t(r, p);
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
// transpose of an 8 x 8 b16 matrix held one row-pair per thread (the MMA C / A fragment layout)
__device__ __forceinline__ uint32_t movtrans(uint32_t a) {
  uint32_t d;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
  return d;
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
               "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(b)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "KDA3_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra KDA3_WAIT_%=;\n"
      "}\n" ::"r"(smem_u32(b)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void st_release(uint32_t* p, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(b)) : "memory");
}
// this thread's earlier cp.async copies arrive on b when complete (counted in b's init count)
__device__ __forceinline__ void cp_async_arrive_noinc(uint64_t* b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::
          "r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(b))
      : "memory");
}
}  // namespace v3

// ---------------------------------------------------------------------------------------- prep
namespace v3 {
// bf16 [64][128] tiles in shared memory: 256-byte rows, 16-byte chunk index XOR (row & 7)
// (conflict-free ldmatrix, row-contiguous per-channel accesses)
__device__ __forceinline__ int swz(int r, int c) {
  return r * 256 + ((((c >> 3) ^ (r & 7))) << 4) + ((c & 7) << 1);
}
constexpr int kTileB = kC * kD * 2;
struct Prep4Smem {
  uint8_t tq[kTileB];   // raw q -> qA = q e^{G - G_m} (bf16) -> AB ([64][kAP] bf16)
  uint8_t tk[kTileB];   // raw k -> kA = k e^{G - G_m}
  uint8_t tg[kTileB];   // raw g -> kB = k e^{G_m - G} -> the inverse's per-warp scratch
  float X[kC][kXP];     // L^T (strict upper) / T^-1 (lower)
  float eGm[4][kD];     // e^{G_m} of each row block's middle row m
  float F[6][kD];       // e^{G_m(a) - G_m(b)}, a > b (pair a (a - 1) / 2 + b)
  float rq[kC], rk[kC], beta[kC];
  float gx[6][kD];      // decay pass: first-half totals, the 4 block middles, the last row
};
__device__ __forceinline__ uint32_t to_tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// e^x as __expf computes it (ex2 of x times log2 e in fp32), without the denormal fix-up of the
// non-ftz ex2 (a compare and two predicated multiplies per call): equal to __expf wherever the
// result is normal, i.e. x log2 e >= -126
constexpr float kLog2e = 1.4426950216293334961f;
__device__ __forceinline__ float ex2_ftz(float y) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(y));
  return r;
}
__device__ __forceinline__ uint32_t scale_bf16x2(uint32_t v, float s0, float s1) {
  __nv_bfloat162 h = *reinterpret_cast<const __nv_bfloat162*>(&v);
  return pack_bf16(__low2float(h) * s0, __high2float(h) * s1);
}
}  // namespace v3

// One CTA of 4 warps per (chunk, head); warp a owns the chunk rows [16a, 16a + 16) for L / M,
// the inverse's diagonal block and W, thread c the channel c in the decay pass. Every operand
// of an MMA is a bf16 tile in shared memory with its decay relative to a row block's middle row
// m: A side x e^{G - G_m(a)}, B side k e^{G_m(b) - G} times e^{G_m(a) - G_m(b)} (<= 1) per channel,
// so no exponent leaves +-8 x the gate's lower bound.
// kFtz (|lower bound| <= 10): the per-row decays e^{+-(G - G_m)} (|exponent| <= 8 |lower bound| < 87)
// are always normal, so they skip the denormal fix-up; the gates' e^x always do (1 + e^x is exactly
// 1 when e^x is denormal)
template <bool kFtz>
__global__ void __launch_bounds__(v3::kPT, 3) kda_chunk_prep3_kernel(
    const bf16* __restrict__ qkv, const bf16* __restrict__ beta_raw, int ld_beta,
    const bf16* __restrict__ g1, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    const int2* __restrict__ chunks, float* __restrict__ scratch, int H, float lower_bound,
    float scale, uint32_t* __restrict__ ready, uint32_t* __restrict__ counters, int n_scan) {
  pdl_trigger();
  // overlap mode (ready != nullptr): launched behind the scan, which waited for the previous
  // kernels and zeroed the flags before it let this grid start; grid (H, chunks), chunk-major
  if (!ready) pdl_wait();
  using namespace v3;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  Prep4Smem& sm = *reinterpret_cast<Prep4Smem*>(smem_raw);
  const int c = ready ? blockIdx.y : blockIdx.x, h = ready ? blockIdx.x : blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31, a = tid >> 5, g = lane >> 2, tq = lane & 3;
  const int2 ch = __ldcg(chunks + c);
  const int t0 = ch.x, n = ch.y;
  const int HD = H * kD;
  const int last = n > 0 ? n - 1 : 0;
  uint8_t* img = reinterpret_cast<uint8_t*>(scratch + ((size_t)c * H + h) * kBlk);
  bf16* oW = reinterpret_cast<bf16*>(img + kImgW);
  bf16* oQh = reinterpret_cast<bf16*>(img + kImgQh);
  bf16* oKd = reinterpret_cast<bf16*>(img + kImgKd);
  bf16* oM = reinterpret_cast<bf16*>(img + kImgM);
  bf16* oAB = reinterpret_cast<bf16*>(img + kImgAB);
  if (tid == 0) *reinterpret_cast<int2*>(img + kImgCh) = ch;
  // ---- P1: raw q, k, g into swizzled tiles (zero rows past the chunk); row norms of q and k
  for (int i = tid; i < kC * (kD / 8); i += kPT) {
    const int r = i >> 4, p = i & 15;
    const bool ok = r < n;
    const size_t row = (size_t)(t0 + (ok ? r : 0));
    const int off = r * 256 + ((p ^ (r & 7)) << 4);
    cp_async16(sm.tq + off, qkv + row * 3 * HD + h * kD + p * 8, ok);
    cp_async16(sm.tk + off, qkv + row * 3 * HD + HD + h * kD + p * 8, ok);
    cp_async16(sm.tg + off, g1 + row * HD + h * kD + p * 8, ok);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  const float amp = expf(__ldg(a_log + h));
  {
    const int rb = 16 * a + (lane & 15);
    float braw = 0.f;
    if (lane < 16 && rb < n) braw = __bfloat162float(beta_raw[(size_t)(t0 + rb) * ld_beta + h]);
    if (lane < 16) sm.beta[rb] = rb < n ? 1.f / (1.f + __expf(-braw)) : 0.f;
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  {
    // lane pair (2 i, 2 i + 1) -> row 16 a + i, channels [64 (lane & 1), + 64)
    const int r = 16 * a + (lane >> 1), half = lane & 1;
    float qq = 0.f, kk = 0.f;
#pragma unroll
    for (int p8 = 0; p8 < 8; ++p8) {
      const int p = 8 * half + p8;
      const int off = r * 256 + ((p ^ (r & 7)) << 4);
      const uint4 qu = *reinterpret_cast<const uint4*>(sm.tq + off);
      const uint4 ku = *reinterpret_cast<const uint4*>(sm.tk + off);
      const bf16* qh = reinterpret_cast<const bf16*>(&qu);
      const bf16* kh = reinterpret_cast<const bf16*>(&ku);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float qv = __bfloat162float(qh[e]), kv = __bfloat162float(kh[e]);
        qq += qv * qv;
        kk += kv * kv;
      }
    }
    qq += __shfl_xor_sync(0xffffffff, qq, 1);
    kk += __shfl_xor_sync(0xffffffff, kk, 1);
    if (half == 0) {
      sm.rq[r] = rsqrtf(qq + 1e-6f);
      sm.rk[r] = rsqrtf(kk + 1e-6f);
    }
  }
  __syncthreads();
  // ---- P2: thread (channel pair c2 = 2 (tid & 63), row half rh = tid / 64): gates and the
  //      cumulative decay of its 32 rows x 2 channels in registers, the first half's totals carried
  //      over through shared memory; then per row qA, kA, kB (tiles, in place) and Qh, Kd (image)
  {
    const int c2 = 2 * (tid & 63), rh = tid >> 6;
    const float2 bias2 = *reinterpret_cast<const float2*>(dt_bias + h * kD + c2);
    float v0[32], v1[32];
    float a0 = 0.f, a1 = 0.f, l0 = 0.f, l1 = 0.f;  // (l: the last row's decay, this half's part)
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int r = 32 * rh + i;
      const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(sm.tg + swz(r, c2));
      if (r < n) {
        a0 += __fdividef(lower_bound,
                         1.f + ex2_ftz(-amp * (__low2float(g2) + bias2.x) * kLog2e));
        a1 += __fdividef(lower_bound,
                         1.f + ex2_ftz(-amp * (__high2float(g2) + bias2.y) * kLog2e));
      }
      v0[i] = a0;
      v1[i] = a1;
      if (r == last) {
        l0 = a0;
        l1 = a1;
      }
    }
    if (rh == 0) {
      sm.gx[0][c2] = a0;
      sm.gx[0][c2 + 1] = a1;
    }
    __syncthreads();
    if (rh == 1) {
      const float t0v = sm.gx[0][c2], t1v = sm.gx[0][c2 + 1];
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        v0[i] += t0v;
        v1[i] += t1v;
      }
      l0 += t0v;
      l1 += t1v;
    }
    // block middles (rows 16 b + 8) and the last row's decay
    sm.gx[1 + 2 * rh][c2] = v0[8];
    sm.gx[1 + 2 * rh][c2 + 1] = v1[8];
    sm.gx[2 + 2 * rh][c2] = v0[24];
    sm.gx[2 + 2 * rh][c2 + 1] = v1[24];
    if ((last >> 5) == rh) {
      sm.gx[5][c2] = l0;
      sm.gx[5][c2 + 1] = l1;
    }
    __syncthreads();
    float gm0[4], gm1[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      gm0[b] = sm.gx[1 + b][c2];
      gm1[b] = sm.gx[1 + b][c2 + 1];
    }
    const float gl0 = sm.gx[5][c2], gl1 = sm.gx[5][c2 + 1];
    if (rh == 0) {
#pragma unroll
      for (int b = 0; b < 4; ++b)
        *reinterpret_cast<float2*>(&sm.eGm[b][c2]) = make_float2(__expf(gm0[b]), __expf(gm1[b]));
      *reinterpret_cast<float2*>(reinterpret_cast<float*>(img) + c2) = make_float2(__expf(gl0), __expf(gl1));
    } else {
#pragma unroll
      for (int a2 = 1; a2 < 4; ++a2)
#pragma unroll
        for (int b = 0; b < a2; ++b)
          *reinterpret_cast<float2*>(&sm.F[a2 * (a2 - 1) / 2 + b][c2]) =
              make_float2(__expf(gm0[a2] - gm0[b]), __expf(gm1[a2] - gm1[b]));
    }
#pragma unroll
    for (int bb = 0; bb < 2; ++bb) {
      // this half's row block 2 rh + bb (constant indices only: the arrays stay in registers)
      const float gb0 = rh ? gm0[2 + bb] : gm0[bb], gb1 = rh ? gm1[2 + bb] : gm1[bb];
      const float eld0 = __expf(gl0 - gb0), eld1 = __expf(gl1 - gb1);   // e^{G_C - G_m}
      const float qs0 = __expf(gb0) * scale, qs1 = __expf(gb1) * scale;
#pragma unroll
      for (int i2 = 0; i2 < 16; ++i2) {
        const int i = 16 * bb + i2, r = 32 * rh + i;
        const int off = swz(r, c2);
        const float x0 = v0[i] - gb0, x1 = v1[i] - gb1;
        float e0, e1, ei0, ei1;
        if constexpr (kFtz) {
          const float y0 = x0 * kLog2e, y1 = x1 * kLog2e;  // ((-x) log2 e = -(x log2 e) exactly)
          e0 = ex2_ftz(y0);
          e1 = ex2_ftz(y1);
          ei0 = ex2_ftz(-y0);
          ei1 = ex2_ftz(-y1);
        } else {
          e0 = __expf(x0);
          e1 = __expf(x1);
          ei0 = __expf(-x0);
          ei1 = __expf(-x1);
        }
        const __nv_bfloat162 q2 = *reinterpret_cast<const __nv_bfloat162*>(sm.tq + off);
        const __nv_bfloat162 k2 = *reinterpret_cast<const __nv_bfloat162*>(sm.tk + off);
        const float rq = sm.rq[r], rk = sm.rk[r];
        const float qn0 = __low2float(q2) * rq, qn1 = __high2float(q2) * rq;
        const float kn0 = __low2float(k2) * rk, kn1 = __high2float(k2) * rk;
        const float qa0 = qn0 * e0, qa1 = qn1 * e1;
        const float kb0 = kn0 * ei0, kb1 = kn1 * ei1;
        *reinterpret_cast<uint32_t*>(sm.tq + off) = pack_bf16(qa0, qa1);
        *reinterpret_cast<uint32_t*>(sm.tk + off) = pack_bf16(kn0 * e0, kn1 * e1);
        *reinterpret_cast<uint32_t*>(sm.tg + off) = pack_bf16(kb0, kb1);
        *reinterpret_cast<uint32_t*>(oQh + r * kHP + c2) = pack_bf16(qa0 * qs0, qa1 * qs1);
        *reinterpret_cast<uint32_t*>(oKd + r * kHP + c2) =
            r < n ? pack_bf16(kb0 * eld0, kb1 * eld1) : 0u;
      }
    }
  }
  __syncthreads();
  // ---- P3: the 10 lower tiles (row block ab, column block b <= ab) of L and M, round-robin over
  //      the warps (BF16 MMA, fp32 accumulation)
  {
    const int arow = lane & 15, acol = (lane >> 4) * 8;
    const int trow = (lane & 7) + ((lane >> 4) << 3), tcol = ((lane >> 3) & 1) << 3;
    for (int tile = a; tile < 10; tile += 4) {
      const int ab = tile >= 6 ? 3 : (tile >= 3 ? 2 : (tile >= 1 ? 1 : 0));
      const int b = tile - ab * (ab + 1) / 2;
      const int a = ab;  // (row block of this tile; shadows the warp index)
      const int ra = 16 * a + g;
      float accL[2][4] = {}, accM[2][4] = {};
      const float* Fp = sm.F[a * (a - 1) / 2 + b];
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t ka[4], qa[4], kb[4];
        ldsm4(ka, sm.tk + swz(16 * a + arow, 16 * ks + acol));
        ldsm4(qa, sm.tq + swz(16 * a + arow, 16 * ks + acol));
        ldsm4(kb, sm.tg + swz(16 * b + trow, 16 * ks + tcol));
        if (b < a) {
          const float2 f0 = *reinterpret_cast<const float2*>(Fp + 16 * ks + 2 * tq);
          const float2 f1 = *reinterpret_cast<const float2*>(Fp + 16 * ks + 8 + 2 * tq);
          kb[0] = scale_bf16x2(kb[0], f0.x, f0.y);
          kb[1] = scale_bf16x2(kb[1], f1.x, f1.y);
          kb[2] = scale_bf16x2(kb[2], f0.x, f0.y);
          kb[3] = scale_bf16x2(kb[3], f1.x, f1.y);
        }
        mma16816(accL[0], ka, kb[0], kb[1]);
        mma16816(accL[1], ka, kb[2], kb[3]);
        mma16816(accM[0], qa, kb[0], kb[1]);
        mma16816(accM[1], qa, kb[2], kb[3]);
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int col = 16 * b + 8 * nt + 2 * tq;
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
          const int row = ra + 8 * hr;
          // L^T in the strict upper triangle (T^-1 fills the lower one in place)
          if (col < row) sm.X[col][row] = accL[nt][2 * hr];
          if (col + 1 < row) sm.X[col + 1][row] = accL[nt][2 * hr + 1];
          const float m0 = col <= row ? accM[nt][2 * hr] * scale : 0.f;
          const float m1 = col + 1 <= row ? accM[nt][2 * hr + 1] * scale : 0.f;
          *reinterpret_cast<__nv_bfloat162*>(oM + row * kAP + col) = __floats2bfloat162_rn(m0, m1);
        }
      }
    }
    // the upper blocks of M: zeros
    const int ncu = (kC - 16 * (a + 1)) / 8;  // 16-byte pieces per row
    for (int i = lane; i < 16 * ncu; i += 32) {
      const int row = 16 * a + i / ncu, col = 16 * (a + 1) + 8 * (i % ncu);
      *reinterpret_cast<uint4*>(oM + row * kAP + col) = make_uint4(0, 0, 0, 0);
    }
  }
  __syncthreads();  // L complete; qA / kB no longer read
  // ---- P5: T^-1 = (I + diag(b) L)^-1 into X's lower triangle (L^T stays in the upper one).
  //      Diagonal block a, lane j < 16 holds column j in registers (forward substitution):
  //        x_i = -b_i sum_{r < i} L[i][r] x_r  (i > j),  x_j = 1,  x_i = 0 (i < j)
  {
    const int base = 16 * a, j = lane & 15;
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = i == j ? 1.f : 0.f;
#pragma unroll
    for (int i = 1; i < 16; ++i) {
      float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
      for (int r = 0; r < i; r += 2) {
        acc0 += sm.X[base + r][base + i] * x[r];
        if (r + 1 < i) acc1 += sm.X[base + r + 1][base + i] * x[r + 1];
      }
      if (i > j) x[i] = -sm.beta[base + i] * (acc0 + acc1);
    }
    if (lane < 16)
#pragma unroll
      for (int i = 1; i < 16; ++i)
        if (i > j) sm.X[base + i][base + j] = x[i];
  }
  __syncthreads();
  // off-diagonal blocks (TF32 mma.sync): warp b walks down its column block,
  //   Y = sum_{b <= c < a} N_ac X_cb  (N_ac = diag(b_a) L_ac, from L^T),  X_ab = -(Y + X'_aa Y)
  // with X'_aa the strictly lower part of X_aa; X_cb (c > b) being the warp's own earlier results.
  {
    const int b = a;
    float* Ys = reinterpret_cast<float*>(sm.tg) + b * 16 * 17;  // per-warp [16][17] scratch
    for (int ar = b + 1; ar < 4; ++ar) {
      const float bg = sm.beta[16 * ar + g], bg8 = sm.beta[16 * ar + g + 8];
      float y[2][4] = {};
      for (int cb = b; cb < ar; ++cb) {
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const int r0 = 8 * kk + tq, r1 = r0 + 4;
          const uint32_t af[4] = {to_tf32(bg * sm.X[16 * cb + r0][16 * ar + g]),
                                  to_tf32(bg8 * sm.X[16 * cb + r0][16 * ar + g + 8]),
                                  to_tf32(bg * sm.X[16 * cb + r1][16 * ar + g]),
                                  to_tf32(bg8 * sm.X[16 * cb + r1][16 * ar + g + 8])};
#pragma unroll
          for (int nn = 0; nn < 2; ++nn) {
            const int jj = 8 * nn + g;
            float x0, x1;
            if (cb == b) {  // X_bb: unit lower triangular
              x0 = r0 == jj ? 1.f : (r0 > jj ? sm.X[16 * b + r0][16 * b + jj] : 0.f);
              x1 = r1 == jj ? 1.f : (r1 > jj ? sm.X[16 * b + r1][16 * b + jj] : 0.f);
            } else {
              x0 = sm.X[16 * cb + r0][16 * b + jj];
              x1 = sm.X[16 * cb + r1][16 * b + jj];
            }
            mma_tf32(y[nn], af, to_tf32(x0), to_tf32(x1));
          }
        }
      }
#pragma unroll
      for (int nn = 0; nn < 2; ++nn)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          Ys[(g + 8 * hf) * 17 + 8 * nn + 2 * tq] = y[nn][2 * hf];
          Ys[(g + 8 * hf) * 17 + 8 * nn + 2 * tq + 1] = y[nn][2 * hf + 1];
        }
      __syncwarp();
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int r0 = 8 * kk + tq, r1 = r0 + 4;
        const uint32_t af[4] = {to_tf32(r0 < g ? sm.X[16 * ar + g][16 * ar + r0] : 0.f),
                                to_tf32(r0 < g + 8 ? sm.X[16 * ar + g + 8][16 * ar + r0] : 0.f),
                                to_tf32(r1 < g ? sm.X[16 * ar + g][16 * ar + r1] : 0.f),
                                to_tf32(r1 < g + 8 ? sm.X[16 * ar + g + 8][16 * ar + r1] : 0.f)};
#pragma unroll
        for (int nn = 0; nn < 2; ++nn)
          mma_tf32(y[nn], af, to_tf32(Ys[r0 * 17 + 8 * nn + g]), to_tf32(Ys[r1 * 17 + 8 * nn + g]));
      }
#pragma unroll
      for (int nn = 0; nn < 2; ++nn)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          sm.X[16 * ar + g + 8 * hf][16 * b + 8 * nn + 2 * tq] = -y[nn][2 * hf];
          sm.X[16 * ar + g + 8 * hf][16 * b + 8 * nn + 2 * tq + 1] = -y[nn][2 * hf + 1];
        }
      __syncwarp();
    }
  }
  __syncthreads();
  // ---- AB = T^-1 diag(b) (bf16): over qA's space (A operand of W) and to the image; own rows
  {
    bf16* ab = reinterpret_cast<bf16*>(sm.tq);  // [64][kAP]
    const int r = 16 * a + (lane >> 1), c0 = 32 * (lane & 1);
    uint32_t w[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const int cA = c0 + 2 * e, cB = cA + 1;
      const float vA = cA < r ? sm.X[r][cA] * sm.beta[cA] : (cA == r ? sm.beta[r] : 0.f);
      const float vB = cB < r ? sm.X[r][cB] * sm.beta[cB] : (cB == r ? sm.beta[r] : 0.f);
      w[e] = pack_bf16(vA, vB);
    }
    uint4* s4 = reinterpret_cast<uint4*>(ab + r * kAP + c0);
    uint4* g4 = reinterpret_cast<uint4*>(oAB + r * kAP + c0);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint4 v = make_uint4(w[4 * e], w[4 * e + 1], w[4 * e + 2], w[4 * e + 3]);
      s4[e] = v;
      g4[e] = v;
    }
  }
  __syncthreads();
  // ---- P6: -W = -AB (k e^G) with k e^G = kA e^{G_m} (per row block, per channel): warp a ->
  //      rows [16a, 16a + 16), two halves of 64 channels; AB is lower triangular, so the k-steps
  //      stop at the row block
  {
    const bf16* ab = reinterpret_cast<const bf16*>(sm.tq);
    const int kr = (lane & 7) + (((lane >> 3) & 1) << 3), nc = (lane >> 4) << 3;
#pragma unroll 1
    for (int hf = 0; hf < 2; ++hf) {
      float acc[8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      for (int ks = 0; ks <= a; ++ks) {
        uint32_t af[4];
        ldsm4(af, ab + (16 * a + (lane & 15)) * kAP + 16 * ks + (lane >> 4) * 8);
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
          const int n0 = 64 * hf + 16 * jp;
          uint32_t bb[4];
          ldsm4t(bb, sm.tk + swz(16 * ks + kr, n0 + nc));
          const float s0 = sm.eGm[ks][n0 + g], s1 = sm.eGm[ks][n0 + 8 + g];
          bb[0] = scale_bf16x2(bb[0], s0, s0);
          bb[1] = scale_bf16x2(bb[1], s0, s0);
          bb[2] = scale_bf16x2(bb[2], s1, s1);
          bb[3] = scale_bf16x2(bb[3], s1, s1);
          mma16816(acc[2 * jp], af, bb[0], bb[1]);
          mma16816(acc[2 * jp + 1], af, bb[2], bb[3]);
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
          const int row = 16 * a + g + 8 * hr, col = 64 * hf + 8 * j + 2 * tq;
          *reinterpret_cast<__nv_bfloat162*>(oW + row * kHP + col) =
              __floats2bfloat162_rn(-acc[j][2 * hr], -acc[j][2 * hr + 1]);
        }
    }
  }
  if (ready) {
    // the image is complete: publish it to the scan; the last block holds the grid until every
    // scan CTA is done (so the kernels behind this one see the scan's outputs)
    __syncthreads();
    if (tid == 0) {
      st_release(ready + (size_t)c * H + h, 1u);
      if (atomicAdd(counters + 1, 1u) == gridDim.x * gridDim.y - 1)
        while (ld_acquire(counters) < (uint32_t)n_scan) __nanosleep(200);
    }
  }
}

// ---------------------------------------------------------------------------------------- scan
namespace v3 {
constexpr int kXP16 = 24;  // exchange rows of up to 16 columns + pad (48 bytes: conflict-free ldmatrix)
template <int COLS>
struct ScanStageT {
  uint8_t img[kImgBytes];  // e^{G_C} | -W | Qh | Kd | M | AB
  bf16 V[kC][COLS + 8];
};
template <int NW, int SN>  // warps per CTA (NW / 2 pairs of SN = 8 or 16 columns)
struct ScanSmemT {
  ScanStageT<NW / 2 * SN> st[2];
  bf16 Sx[NW / 2][kD][kXP16];  // the pairs' bf16 states (read before the pair barrier, rewritten after)
  bf16 Ux[NW / 2][kC][kXP16];  // the pairs' bf16 U (rewritten after the next chunk's CTA barrier)
  uint64_t bar[2];
  uint64_t ebar[2];  // producer form: every compute warp is done with the stage
};
}  // namespace v3

namespace v3 {
// One chunk of the scan for the warp holding state rows [64 HR, + 64) / chunk rows [32 HR, + 32) of
// its pair's 16 columns (HR at compile time: the lower-triangular loop bounds are constants). Each
// phase's A fragments are loaded together ahead of its MMAs.
template <int HR, int NI, class Stage>
__device__ __forceinline__ void scan_chunk_step(
    float (&s)[4][NI][4], const bf16 (*Sx)[kXP16], bf16 (*Ux)[kXP16], const Stage& t,
    const bf16* tW, const bf16* tQh, const bf16* tKd, const bf16* tM, const bf16* tAB,
    const float* eG, bf16* __restrict__ o, int HD, int h, int col0, int wc, int lane, int pbar) {
  // chunk-row tiles of this warp (2 of 4, balanced over the triangular AB / M work): HR 0 {0, 3},
  // HR 1 {1, 2}
  constexpr int kMt[2] = {HR == 0 ? 0 : 1, HR == 0 ? 3 : 2};
  const int2 ch = *reinterpret_cast<const int2*>(t.img + kImgCh);
  const int g = lane >> 2, tq = lane & 3;
  const int arow = lane & 15, acol = (lane >> 4) * 8;
  const int kr = (lane & 7) + (((lane >> 3) & 1) << 3), nc = NI == 2 ? (lane >> 4) << 3 : 0;
  const int trow = (lane & 7) + ((lane >> 4) << 3), tcol = ((lane >> 3) & 1) << 3;
  // ---- the pair's state as B fragments over all 128 rows (K); -W's A fragments
  uint32_t sb[8][2 * NI], wf[8][2][4];
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) ldsm_bt<NI>(sb[ks], &Sx[16 * ks + kr][nc]);
#pragma unroll
  for (int ks = 0; ks < 8; ++ks)
#pragma unroll
    for (int m2 = 0; m2 < 2; ++m2) ldsm4(wf[ks][m2], tW + (16 * kMt[m2] + arow) * kHP + 16 * ks + acol);
  // ---- U rows [32 HR, + 32) = AB V + (-W) S
  float u[2][NI][4];
#pragma unroll
  for (int m2 = 0; m2 < 2; ++m2)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) u[m2][ni][0] = u[m2][ni][1] = u[m2][ni][2] = u[m2][ni][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks <= kMt[1]; ++ks) {
    uint32_t vb[2 * NI];
    ldsm_bt<NI>(vb, &t.V[16 * ks + kr][wc + nc]);
#pragma unroll
    for (int m2 = 0; m2 < 2; ++m2) {
      const int mt = kMt[m2];
      if (ks > mt) continue;
      uint32_t af[4];
      ldsm4(af, tAB + (16 * mt + arow) * kAP + 16 * ks + acol);
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) mma16816(u[m2][ni], af, vb[2 * ni], vb[2 * ni + 1]);
    }
  }
#pragma unroll
  for (int ks = 0; ks < 8; ++ks)
#pragma unroll
    for (int m2 = 0; m2 < 2; ++m2)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) mma16816(u[m2][ni], wf[ks][m2], sb[ks][2 * ni], sb[ks][2 * ni + 1]);
  // ---- exchange U within the pair, then U as B fragments over all 64 rows
#pragma unroll
  for (int m2 = 0; m2 < 2; ++m2)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int e = 0; e < 4; e += 2) {
        const int row = 16 * kMt[m2] + g + 8 * (e >> 1), col = 8 * ni + 2 * tq;
        *reinterpret_cast<uint32_t*>(&Ux[row][col]) = pack_bf16(u[m2][ni][e], u[m2][ni][e + 1]);
      }
  uint32_t qf[8][2][4];
#pragma unroll
  for (int ks = 0; ks < 8; ++ks)
#pragma unroll
    for (int m2 = 0; m2 < 2; ++m2) ldsm4(qf[ks][m2], tQh + (16 * kMt[m2] + arow) * kHP + 16 * ks + acol);
  asm volatile("bar.sync %0, 64;\n" ::"r"(pbar) : "memory");
  uint32_t ub[4][2 * NI];
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) ldsm_bt<NI>(ub[ks], &Ux[16 * ks + kr][nc]);
  // ---- O rows [32 HR, + 32) = Qh S + M U (rows < n)
  {
    float acc[2][NI][4];
#pragma unroll
    for (int m2 = 0; m2 < 2; ++m2)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) acc[m2][ni][0] = acc[m2][ni][1] = acc[m2][ni][2] = acc[m2][ni][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
#pragma unroll
      for (int m2 = 0; m2 < 2; ++m2)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma16816(acc[m2][ni], qf[ks][m2], sb[ks][2 * ni], sb[ks][2 * ni + 1]);
#pragma unroll
    for (int ks = 0; ks <= kMt[1]; ++ks)
#pragma unroll
      for (int m2 = 0; m2 < 2; ++m2) {
        const int mt = kMt[m2];
        if (ks > mt) continue;
        uint32_t af[4];
        ldsm4(af, tM + (16 * mt + arow) * kAP + 16 * ks + acol);
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma16816(acc[m2][ni], af, ub[ks][2 * ni], ub[ks][2 * ni + 1]);
      }
    const size_t obase = (size_t)ch.x * HD + h * kD + col0 + wc;
#pragma unroll
    for (int m2 = 0; m2 < 2; ++m2)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int row = 16 * kMt[m2] + g + 8 * hf;
          if (row < ch.y)
            *reinterpret_cast<__nv_bfloat162*>(o + obase + (size_t)row * HD + 8 * ni + 2 * tq) =
                __floats2bfloat162_rn(acc[m2][ni][2 * hf], acc[m2][ni][2 * hf + 1]);
        }
  }
  // ---- state rows [64 HR, + 64) = e^{G_C} S + Kd^T U (Kd^T fragments by ldmatrix.trans of Kd)
  uint32_t kf[4][4][4];
#pragma unroll
  for (int ks = 0; ks < 4; ++ks)
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) ldsm4t(kf[ks][mi], tKd + (16 * ks + trow) * kHP + 64 * HR + 16 * mi + tcol);
#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
    const float e0 = eG[64 * HR + 16 * mi + g], e1 = eG[64 * HR + 16 * mi + g + 8];
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      s[mi][ni][0] *= e0;
      s[mi][ni][1] *= e0;
      s[mi][ni][2] *= e1;
      s[mi][ni][3] *= e1;
    }
  }
#pragma unroll
  for (int ks = 0; ks < 4; ++ks)
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) mma16816(s[mi][ni], kf[ks][mi], ub[ks][2 * ni], ub[ks][2 * ni + 1]);
}
}  // namespace v3

// grid (B, H, 256 / NW), NW warps: warp pair p owns the CTA's columns [16p, 16p + 16); warp hr of the pair
// owns state rows [64 hr, + 64) and chunk rows [32 hr, + 32). Per chunk the pair exchanges its bf16
// state (published by the chunk's CTA barrier) and its U halves (pair barrier).
template <int NW, int SN>
__global__ void __launch_bounds__(NW * 32, 1) kda_chunk_scan3_kernel(
    bf16* __restrict__ o, const bf16* __restrict__ qkv, const float* __restrict__ scratch,
    float* __restrict__ states32, const int2* __restrict__ chunks,
    const int* __restrict__ seq_chunks, float* __restrict__ state, const KdaSeqMeta meta, int H,
    const KdaSnap* __restrict__ snaps, int n_snaps, uint32_t* __restrict__ ready,
    uint32_t* __restrict__ counters, int n_ready, int bsel, int nseq) {
  if (!ready) pdl_trigger();
  pdl_wait();
  using namespace v3;
  constexpr int kSCols = NW / 2 * SN, kSThreads = NW * 32, NI = SN / 8;
  // overlap mode (ready != nullptr): the prep grid runs behind this one on the SMs it leaves free and
  // flags each (chunk, head) image when it is written. Having waited for the previous kernels, zero
  // the flags and counters, then let the prep grid start.
  if (ready) {
    const int cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const int ncta = gridDim.x * gridDim.y * gridDim.z;
    for (int i = cta * kSThreads + (int)threadIdx.x; i < n_ready; i += ncta * kSThreads) ready[i] = 0u;
    if (cta == 0 && threadIdx.x < 2) counters[threadIdx.x] = 0u;
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      pdl_trigger();
    }
  }
  using ScanStage = ScanStageT<kSCols>;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  ScanSmemT<NW, SN>& sm = *reinterpret_cast<ScanSmemT<NW, SN>*>(smem_raw);
  // bsel 1 / 2: this grid covers only the sequences of more / at most two chunks (blockIdx.x-th)
  int b = blockIdx.x;
  if (bsel) {
    for (int i = 0, k = blockIdx.x; i < nseq; ++i)
      if ((__ldcg(seq_chunks + i + 1) - __ldcg(seq_chunks + i) > 2) == (bsel == 1) && k-- == 0) {
        b = i;
        break;
      }
  }
  const int h = blockIdx.y, col0 = blockIdx.z * kSCols, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int pr = warp >> 1, hr = warp & 1;
  const int wc = SN * pr;  // the pair's first column within the CTA
  const int HD = H * kD;
  if (tid == 0) {
    mbar_init(&sm.bar[0], 1);
    mbar_init(&sm.bar[1], 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  const int slot = meta.slots[b];
  const bool has_state = meta.has_state[b];
  float* S = state + (size_t)slot * meta.state_ss + (size_t)h * kD * kD;
  const int c0 = __ldcg(seq_chunks + b), c1 = __ldcg(seq_chunks + b + 1);
  // chunk c's tiles into stage buf: zeros for V rows past the chunk (generic stores, published by
  // the barrier before the stage is read), then the tile image and the V rows by bulk copies
  auto issue = [&](int c, int buf, int2 ch) {
    ScanStage& t = sm.st[buf];
    if (tid == 0) {
      if (ready) {  // the prep block of (c, h) published its image
        const uint32_t* f = ready + (size_t)c * H + h;
        while (ld_acquire(f) == 0u) __nanosleep(64);
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      mbar_expect_tx(&sm.bar[buf], (uint32_t)kImgBytes);
      bulk_g2s(t.img, scratch + ((size_t)c * H + h) * kBlk, kImgBytes, &sm.bar[buf]);
    }
    // V rows (zeros past the chunk): one cp.async group per chunk
    for (int i = tid; i < kC * (kSCols / 8); i += kSThreads) {
      const int r = i / (kSCols / 8), q = (i % (kSCols / 8)) * 8;
      const bool ok = r < ch.y;
      cp_async16(&t.V[r][q], qkv + (size_t)(ch.x + (ok ? r : 0)) * 3 * HD + 2 * HD + h * kD + col0 + q,
                 ok);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  // state fragments: s[mi][ni][e] = S[64 hr + 16 mi + g + 8 (e >> 1)][col0 + wc + 8 ni + 2 tq + (e & 1)]
  float s[4][NI][4];
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int e = 0; e < 4; e += 2) {
        const int row = 64 * hr + 16 * mi + g + 8 * (e >> 1), col = col0 + wc + 8 * ni + 2 * tq;
        float2 v = make_float2(0.f, 0.f);
        if (has_state) v = *reinterpret_cast<const float2*>(S + (size_t)row * kD + col);
        s[mi][ni][e] = v.x;
        s[mi][ni][e + 1] = v.y;
      }
  // this warp's half of the state into the pair's exchange buffer
  auto put_state = [&]() {
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
          const int row = 64 * hr + 16 * mi + g + 8 * (e >> 1), col = 8 * ni + 2 * tq;
          *reinterpret_cast<uint32_t*>(&sm.Sx[pr][row][col]) =
              pack_bf16(s[mi][ni][e], s[mi][ni][e + 1]);
        }
  };
  put_state();
  __syncthreads();  // barriers initialized; the states written
  int2 ch = c0 < c1 ? __ldcg(chunks + c0) : make_int2(0, 0);
  if (c0 < c1) issue(c0, 0, ch);
  int2 ch_n1 = c0 + 1 < c1 ? __ldcg(chunks + c0 + 1) : make_int2(0, 0);
  // the next chunk in [c, c1) whose initial state a prefix-reuse snapshot copies (c1: none), looked
  // up once per snapshot instead of a global load per chunk on the loop's critical path
  auto next_snap = [&](int c) {
    int nx = c1;
    for (int i = 0; i < n_snaps; ++i) {
      const int sc = snaps[i].chunk;
      if (sc >= c && sc < nx) nx = sc;
    }
    return nx;
  };
  int snap_c = n_snaps > 0 ? next_snap(c0) : c1;
  uint32_t ph = 0;  // bit buf: the parity stage buf waits for next
  const int arow = lane & 15, acol = (lane >> 4) * 8;
  const int kr = (lane & 7) + (((lane >> 3) & 1) << 3), nc = (lane >> 4) << 3;
  const int trow = (lane & 7) + ((lane >> 4) << 3), tcol = ((lane >> 3) & 1) << 3;
  const int pbar = 1 + pr;  // the pair's named barrier
  for (int c = c0; c < c1; ++c) {
    const int buf = (c - c0) & 1;
    // this chunk's V rows (the only cp.async group in flight: the next one is issued below)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    mbar_wait(&sm.bar[buf], (ph >> buf) & 1u);
    ph ^= 1u << buf;
    __syncthreads();  // stage buf and the pairs' states published; the other stage is free
    if (c + 1 < c1) issue(c + 1, buf ^ 1, ch_n1);
    const ScanStage& t = sm.st[buf];
    const float* eG = reinterpret_cast<const float*>(t.img);
    const bf16* tW = reinterpret_cast<const bf16*>(t.img + kImgW);    // -W
    const bf16* tQh = reinterpret_cast<const bf16*>(t.img + kImgQh);
    const bf16* tKd = reinterpret_cast<const bf16*>(t.img + kImgKd);
    const bf16* tM = reinterpret_cast<const bf16*>(t.img + kImgM);
    const bf16* tAB = reinterpret_cast<const bf16*>(t.img + kImgAB);
    const bf16(*Sx)[kXP16] = sm.Sx[pr];
    bf16(*Ux)[kXP16] = sm.Ux[pr];
    const int2 ch_n2 = c + 2 < c1 ? __ldcg(chunks + c + 2) : make_int2(0, 0);  // (used at the loop end)
    // the chunk's initial state where a prefix-reuse snapshot copies it
    if (c == snap_c) {
      snap_c = next_snap(c + 1);
      {
        float* Sc = states32 + ((size_t)c * H + h) * kD * kD;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
          for (int ni = 0; ni < NI; ++ni)
#pragma unroll
            for (int e = 0; e < 4; e += 2) {
              const int row = 64 * hr + 16 * mi + g + 8 * (e >> 1);
              const int col = col0 + wc + 8 * ni + 2 * tq;
              *reinterpret_cast<float2*>(Sc + (size_t)row * kD + col) =
                  make_float2(s[mi][ni][e], s[mi][ni][e + 1]);
            }
      }
    }
    if (hr == 0)
      scan_chunk_step<0, NI>(s, Sx, Ux, t, tW, tQh, tKd, tM, tAB, eG, o, HD, h, col0, wc, lane, pbar);
    else
      scan_chunk_step<1, NI>(s, Sx, Ux, t, tW, tQh, tKd, tM, tAB, eG, o, HD, h, col0, wc, lane, pbar);
    put_state();  // (both warps of the pair read the old state before the pair barrier) published
                  // by the next chunk's CTA barrier
    ch = ch_n1;
    ch_n1 = ch_n2;
  }
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int e = 0; e < 4; e += 2) {
        const int row = 64 * hr + 16 * mi + g + 8 * (e >> 1), col = col0 + wc + 8 * ni + 2 * tq;
        *reinterpret_cast<float2*>(S + (size_t)row * kD + col) =
            make_float2(s[mi][ni][e], s[mi][ni][e + 1]);
      }
  if (ready) {  // outputs and states written: count this CTA done (the prep grid's last block waits)
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      atomicAdd(counters, 1u);
    }
  }
}


// Producer form: NW compute warps (pairs as in the plain form) and one producer warp. The producer
// waits for a stage to be free (every compute warp arrived on its empty barrier), for the chunk's
// prep flag in the overlap, then issues the image's bulk copy and the V rows (cp.async completing
// on the stage's full barrier). The compute warps never issue loads or poll flags, and sync per
// pair (the states' and U's exchanges) instead of per CTA. Bitwise the plain form.
template <int NW, int SN>
__global__ void __launch_bounds__(NW * 32 + 32, 1) kda_chunk_scan3p_kernel(
    bf16* __restrict__ o, const bf16* __restrict__ qkv, const float* __restrict__ scratch,
    float* __restrict__ states32, const int2* __restrict__ chunks,
    const int* __restrict__ seq_chunks, float* __restrict__ state, const KdaSeqMeta meta, int H,
    const KdaSnap* __restrict__ snaps, int n_snaps, uint32_t* __restrict__ ready,
    uint32_t* __restrict__ counters, int n_ready, int bsel, int nseq) {
  if (!ready) pdl_trigger();
  pdl_wait();
  using namespace v3;
  constexpr int kSCols = NW / 2 * SN, kCThreads = NW * 32, kSThreads = kCThreads + 32, NI = SN / 8;
  if (ready) {  // (as the plain form: zero the flags and counters, then let the prep grid start)
    const int cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const int ncta = gridDim.x * gridDim.y * gridDim.z;
    for (int i = cta * kSThreads + (int)threadIdx.x; i < n_ready; i += ncta * kSThreads) ready[i] = 0u;
    if (cta == 0 && threadIdx.x < 2) counters[threadIdx.x] = 0u;
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      pdl_trigger();
    }
  }
  using ScanStage = ScanStageT<kSCols>;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  ScanSmemT<NW, SN>& sm = *reinterpret_cast<ScanSmemT<NW, SN>*>(smem_raw);
  int b = blockIdx.x;
  if (bsel) {
    for (int i = 0, k = blockIdx.x; i < nseq; ++i)
      if ((__ldcg(seq_chunks + i + 1) - __ldcg(seq_chunks + i) > 2) == (bsel == 1) && k-- == 0) {
        b = i;
        break;
      }
  }
  const int h = blockIdx.y, col0 = blockIdx.z * kSCols, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int pr = warp >> 1, hr = warp & 1;
  const int wc = SN * pr;
  const int HD = H * kD;
  if (tid == 0) {
    mbar_init(&sm.bar[0], 1 + 32);  // the producer's expect_tx arrive + its 32 lanes' cp.async arrives
    mbar_init(&sm.bar[1], 1 + 32);
    mbar_init(&sm.ebar[0], NW);
    mbar_init(&sm.ebar[1], NW);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  const int slot = meta.slots[b];
  const bool has_state = meta.has_state[b];
  float* S = state + (size_t)slot * meta.state_ss + (size_t)h * kD * kD;
  const int c0 = __ldcg(seq_chunks + b), c1 = __ldcg(seq_chunks + b + 1);
  float s[4][NI][4];
  auto put_state = [&]() {
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
          const int row = 64 * hr + 16 * mi + g + 8 * (e >> 1), col = 8 * ni + 2 * tq;
          *reinterpret_cast<uint32_t*>(&sm.Sx[pr][row][col]) =
              pack_bf16(s[mi][ni][e], s[mi][ni][e + 1]);
        }
  };
  if (warp < NW) {
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
          const int row = 64 * hr + 16 * mi + g + 8 * (e >> 1), col = col0 + wc + 8 * ni + 2 * tq;
          float2 v = make_float2(0.f, 0.f);
          if (has_state) v = *reinterpret_cast<const float2*>(S + (size_t)row * kD + col);
          s[mi][ni][e] = v.x;
          s[mi][ni][e + 1] = v.y;
        }
    put_state();
  }
  __syncthreads();  // barriers initialized; the states written
  if (warp == NW) {
    // ---- producer
    uint32_t eph = 0;
    for (int c = c0; c < c1; ++c) {
      const int buf = (c - c0) & 1;
      if (c >= c0 + 2) {  // the stage held chunk c - 2: every compute warp is done with it
        mbar_wait(&sm.ebar[buf], (eph >> buf) & 1u);
        eph ^= 1u << buf;
      }
      const int2 ch = __ldcg(chunks + c);
      ScanStage& t = sm.st[buf];
      if (lane == 0) {
        if (ready) {  // the prep block of (c, h) published its image
          const uint32_t* f = ready + (size_t)c * H + h;
          while (ld_acquire(f) == 0u) __nanosleep(64);
          asm volatile("fence.proxy.async.global;\n" ::: "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        mbar_expect_tx(&sm.bar[buf], (uint32_t)kImgBytes);
        bulk_g2s(t.img, scratch + ((size_t)c * H + h) * kBlk, kImgBytes, &sm.bar[buf]);
      }
      // V rows (zeros past the chunk)
      for (int i = lane; i < kC * (kSCols / 8); i += 32) {
        const int r = i / (kSCols / 8), q = (i % (kSCols / 8)) * 8;
        const bool ok = r < ch.y;
        cp_async16(&t.V[r][q], qkv + (size_t)(ch.x + (ok ? r : 0)) * 3 * HD + 2 * HD + h * kD + col0 + q,
                   ok);
      }
      cp_async_arrive_noinc(&sm.bar[buf]);
    }
  } else {
    // ---- compute warps
    auto next_snap = [&](int c) {
      int nx = c1;
      for (int i = 0; i < n_snaps; ++i) {
        const int sc = snaps[i].chunk;
        if (sc >= c && sc < nx) nx = sc;
      }
      return nx;
    };
    int snap_c = n_snaps > 0 ? next_snap(c0) : c1;
    uint32_t ph = 0;
    const int pbar = 1 + pr;  // the pair's named barrier
    for (int c = c0; c < c1; ++c) {
      const int buf = (c - c0) & 1;
      mbar_wait(&sm.bar[buf], (ph >> buf) & 1u);
      ph ^= 1u << buf;
      asm volatile("bar.sync %0, 64;\n" ::"r"(pbar) : "memory");  // the pair's states published
      const ScanStage& t = sm.st[buf];
      const float* eG = reinterpret_cast<const float*>(t.img);
      const bf16* tW = reinterpret_cast<const bf16*>(t.img + kImgW);
      const bf16* tQh = reinterpret_cast<const bf16*>(t.img + kImgQh);
      const bf16* tKd = reinterpret_cast<const bf16*>(t.img + kImgKd);
      const bf16* tM = reinterpret_cast<const bf16*>(t.img + kImgM);
      const bf16* tAB = reinterpret_cast<const bf16*>(t.img + kImgAB);
      const bf16(*Sx)[kXP16] = sm.Sx[pr];
      bf16(*Ux)[kXP16] = sm.Ux[pr];
      if (c == snap_c) {  // the chunk's initial state where a prefix-reuse snapshot copies it
        snap_c = next_snap(c + 1);
        float* Sc = states32 + ((size_t)c * H + h) * kD * kD;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
          for (int ni = 0; ni < NI; ++ni)
#pragma unroll
            for (int e = 0; e < 4; e += 2) {
              const int row = 64 * hr + 16 * mi + g + 8 * (e >> 1);
              const int col = col0 + wc + 8 * ni + 2 * tq;
              *reinterpret_cast<float2*>(Sc + (size_t)row * kD + col) =
                  make_float2(s[mi][ni][e], s[mi][ni][e + 1]);
            }
      }
      if (hr == 0)
        scan_chunk_step<0, NI>(s, Sx, Ux, t, tW, tQh, tKd, tM, tAB, eG, o, HD, h, col0, wc, lane, pbar);
      else
        scan_chunk_step<1, NI>(s, Sx, Ux, t, tW, tQh, tKd, tM, tAB, eG, o, HD, h, col0, wc, lane, pbar);
      put_state();
      __syncwarp();
      if (lane == 0) mbar_arrive(&sm.ebar[buf]);  // this warp is done with the stage
    }
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
          const int row = 64 * hr + 16 * mi + g + 8 * (e >> 1), col = col0 + wc + 8 * ni + 2 * tq;
          *reinterpret_cast<float2*>(S + (size_t)row * kD + col) =
              make_float2(s[mi][ni][e], s[mi][ni][e + 1]);
        }
  }
  if (ready) {  // outputs and states written: count this CTA done (the prep grid's last block waits)
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      atomicAdd(counters, 1u);
    }
  }
}

size_t kda_chunk3_block_floats() { return v3::kBlk; }

void kda_chunked3(bf16* o, const bf16* qkv, const bf16* beta_raw, int ld_beta, const bf16* g1,
                  const float* a_log, const float* dt_bias, float* state, const KdaSeqMeta& meta,
                  const int2* chunks, const int* seq_chunks, int num_chunks, float* scratch, int B,
                  int H, float lower_bound, cudaStream_t st, const KdaSnap* snaps, int n_snaps,
                  int b_long) {
  using namespace v3;
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_prep3_kernel<false>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)sizeof(Prep4Smem)));
    CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_prep3_kernel<true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)sizeof(Prep4Smem)));
    CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_scan3_kernel<8, 16>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)sizeof(ScanSmemT<8, 16>)));
    CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_scan3_kernel<4, 16>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)sizeof(ScanSmemT<4, 16>)));
    CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_scan3_kernel<8, 8>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)sizeof(ScanSmemT<8, 8>)));
    CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_scan3_kernel<4, 8>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)sizeof(ScanSmemT<4, 8>)));
    CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_scan3p_kernel<4, 8>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)sizeof(ScanSmemT<4, 8>)));
    CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_scan3p_kernel<4, 16>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)sizeof(ScanSmemT<4, 16>)));
    configured[dev] = true;
  }
  const float scale = rsqrtf((float)kD);
  const bool ftz = lower_bound >= -10.f;
  auto launch_prep = [&](dim3 grid, uint32_t* rdy, uint32_t* cnt, int nsc) {
    if (ftz)
      launch_pdl(kda_chunk_prep3_kernel<true>, grid, dim3(kPT), sizeof(Prep4Smem), st, qkv, beta_raw,
                 ld_beta, g1, a_log, dt_bias, chunks, scratch, H, lower_bound, scale, rdy, cnt, nsc);
    else
      launch_pdl(kda_chunk_prep3_kernel<false>, grid, dim3(kPT), sizeof(Prep4Smem), st, qkv,
                 beta_raw, ld_beta, g1, a_log, dt_bias, chunks, scratch, H, lower_bound, scale, rdy,
                 cnt, nsc);
  };
  float* states32 = scratch + (size_t)num_chunks * H * kBlk;
  static int sms = 0;
  if (!sms) CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  // Prep / scan overlap: the scan in its 32-column form (4 CTAs per head, one per SM) starts first;
  // the prep grid follows on the SMs it leaves free and flags each image, the scan's CTAs wait for
  // their chunk's flag. Prep is latency-bound at 3 CTAs per SM: alone it takes about as long as the
  // scan, side by side the step takes about as long as prep on the remaining SMs. Needs PDL (the prep
  // grid starts when every scan CTA let it) and every scan CTA resident with SMs to spare.
  // (b_long < B: only the long sequences' scan runs beside prep; the short ones (one or two chunks,
  // e.g. attention DP's one-token dummies of idle ranks) take a second scan grid behind prep)
  const int n_long = b_long > 0 && b_long < B ? b_long : B;
  // the scan's 4-warp forms with a producer warp (kda_chunk_scan3p_kernel; else the plain form)
  constexpr bool kProducer = true;
  const int n_ovl = n_long * H * 4;
  const bool overlap = n_ovl + 64 <= sms && num_chunks * H >= 2 * sms;
  if (overlap) {
    uint32_t* ready = reinterpret_cast<uint32_t*>(states32 + (size_t)num_chunks * H * kD * kD);
    uint32_t* counters = ready + ((num_chunks * H + 31) & ~31);
    const int bsel = n_long < B ? 1 : 0;
    if (kProducer)
      launch_pdl(kda_chunk_scan3p_kernel<4, 16>, dim3(n_long, H, 4), dim3(160),
                 sizeof(ScanSmemT<4, 16>), st, o, qkv, scratch, states32, chunks, seq_chunks, state,
                 meta, H, snaps, n_snaps, ready, counters, num_chunks * H, bsel, B);
    else
      launch_pdl(kda_chunk_scan3_kernel<4, 16>, dim3(n_long, H, 4), dim3(128),
                 sizeof(ScanSmemT<4, 16>), st, o, qkv, scratch, states32, chunks, seq_chunks, state,
                 meta, H, snaps, n_snaps, ready, counters, num_chunks * H, bsel, B);
    launch_prep(dim3(H, num_chunks), ready, counters, n_ovl);
    if (n_long < B) {
      const int ns = B - n_long;
      if (ns * H * 4 <= sms)
        launch_pdl(kda_chunk_scan3_kernel<4, 16>, dim3(ns, H, 4), dim3(128),
                   sizeof(ScanSmemT<4, 16>), st, o, qkv, scratch, states32, chunks, seq_chunks,
                   state, meta, H, snaps, n_snaps, nullptr, nullptr, 0, 2, B);
      else
        launch_pdl(kda_chunk_scan3_kernel<8, 16>, dim3(ns, H, 2), dim3(256),
                   sizeof(ScanSmemT<8, 16>), st, o, qkv, scratch, states32, chunks, seq_chunks,
                   state, meta, H, snaps, n_snaps, nullptr, nullptr, 0, 2, B);
    }
    return;
  }
  launch_prep(dim3(num_chunks, H), nullptr, nullptr, 0);
  // Column split of the scan (columns evolve independently): CTAs of NW warps, pairs of SN columns.
  // A warp's per-chunk chain (U, the state update, O) shrinks with its columns, so the narrowest form
  // whose CTAs still fit one wave on the SMs: 16 columns (8 CTAs per head, pairs of 8), else 32 (4 per
  // head, pairs of 16), else 64 (2 per head).
  // (b_long: only the sequences of more than a chunk or two set the critical path; attention DP's
  // idle ranks add one-token dummy sequences whose CTAs finish within a chunk)
  const int Bf = b_long > 0 && b_long < B ? b_long : B;
  const int fm = Bf * H * 8 <= sms ? 48 : (Bf * H * 4 <= sms ? 416 : 816);
#define KDA3_SCAN(NW, SN)                                                                          \
  launch_pdl(kda_chunk_scan3_kernel<NW, SN>, dim3(B, H, kD / (NW / 2 * SN)), dim3(NW * 32),         \
             sizeof(ScanSmemT<NW, SN>), st, o, qkv, scratch, states32, chunks, seq_chunks, state,   \
             meta, H, snaps, n_snaps, nullptr, nullptr, 0, 0, B)
#define KDA3_SCANP(NW, SN)                                                                         \
  launch_pdl(kda_chunk_scan3p_kernel<NW, SN>, dim3(B, H, kD / (NW / 2 * SN)), dim3(NW * 32 + 32),   \
             sizeof(ScanSmemT<NW, SN>), st, o, qkv, scratch, states32, chunks, seq_chunks, state,   \
             meta, H, snaps, n_snaps, nullptr, nullptr, 0, 0, B)
  if (kProducer && (fm == 48 || fm == 416)) {
    if (fm == 48)
      KDA3_SCANP(4, 8);
    else
      KDA3_SCANP(4, 16);
    return;
  }
  switch (fm) {
    case 48: KDA3_SCAN(4, 8); break;
    case 416: KDA3_SCAN(4, 16); break;
    default: KDA3_SCAN(8, 16); break;
  }
#undef KDA3_SCAN
#undef KDA3_SCANP
}

}  // namespace k
}  // namespace glm
